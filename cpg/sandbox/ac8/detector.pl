#!/usr/bin/perl
# detector.pl -- the AC-8 logging detector (egress-spec3 §8.1). Runs in a container on the scenario network ONLY.
# It accepts TCP on 80, 443 and 3128, writes one `conn` record per connection and closes without forwarding.
# Its only writable path is /detector (the host's <auditDir>/detector/):
#   conn.jsonl      {"ts","srcAddr","srcPort","dstAddr","dstPort","bytesOut","bytesIn"} -- ts = the accept time, bytesIn = bytes the
#                   peer sent (read for at most 0.3 s), bytesOut = bytes this detector sent (always 0: it never answers)
#   start.marker    written on SIGUSR1: {"marker":"start","ts":...}   (the window opens: before the first `policy` record)
#   flush.marker    written on SIGUSR2: {"marker":"flush","ts":...,"residue":N}   (after the last record). The detector first DRAINS: it
#                   accepts and logs until no listener has a queued connection and none is still being read (the kernel's accept queue,
#                   read from /proc/net/tcp, included), bounded by 30 s of wall-clock, not by a count; N is what was still unlogged when
#                   it stopped (0 on a clean drain; window.py fails the verdict on any other value).
#   probe.jsonl     the harness's own positive-control connections (netprobe.pl): a connection whose first bytes are `AC8PROBE:<nonce>\n`
#                   with <nonce> listed in /detector/probe-tokens (`<nonce> <label>` per line, one per probe target), SINGLE-USE. The record is
#                   the conn record plus "probe":"<label>" (label = component/phase). They are NOT `conn` records, so the probe that proves the
#                   detector's ports accept (P-a's positive control) does not fail the zero-`conn` verdict; assertions.py attributes every one
#                   to a (component, phase) and counts them against what the probes report sending. A token-less, wrong or replayed nonce
#                   is a `conn`.
#   ready           written once the listeners are up
# A marker is written once (O_EXCL); a second signal changes nothing. On SIGTERM it drains once more, writes final.marker
#   {"marker":"final","ts":...,"residue":N} (N: what that drain left unlogged; window.py requires it present and 0) and exits 0 (the harness
#   stops the detector with `docker stop -t` longer than the 30 s drain bound and checks the exit status before it judges). Test hooks (never set in the container): AC8_DET_PORTS=<p,..>, AC8_DET_ADDR=<ip>, AC8_DET_HOLD, AC8_DET_DRAIN_MAX (Docker-free tests on loopback).
use strict; use warnings;
use Fcntl; use IO::Socket::INET; use IO::Select; use Time::HiRes qw(time sleep); use POSIX qw(strftime);
$| = 1;
my $dir = $ARGV[0] // '/detector';
my @ports = $ENV{AC8_DET_PORTS} ? split(/,/, $ENV{AC8_DET_PORTS}) : (80, 443, 3128);
my $addr = $ENV{AC8_DET_ADDR} // '0.0.0.0';
my $HOLD = $ENV{AC8_DET_HOLD} // 0.3;          # how long one connection is read before it is logged (test hook AC8_DET_HOLD)
my $DRAIN_MAX = $ENV{AC8_DET_DRAIN_MAX} // 30;  # wall-clock bound of a flush / termination drain (test hook AC8_DET_DRAIN_MAX)
my ($want_start, $want_flush, $want_term) = (0, 0, 0);
$SIG{USR1} = sub { $want_start = 1 };
$SIG{USR2} = sub { $want_flush = 1 };
$SIG{TERM} = sub { $want_term = 1 };
sub ts { my $t = shift // time; strftime('%Y-%m-%dT%H:%M:%S', gmtime($t)) . sprintf('.%03dZ', ($t - int($t)) * 1000) }
my $sel = IO::Select->new; my $lsel = IO::Select->new; my @listeners;
for my $p (@ports) {
  my $l = IO::Socket::INET->new(Listen => 128, LocalAddr => $addr, LocalPort => $p, Proto => 'tcp', ReuseAddr => 1, Blocking => 0)
    or die "detector: cannot listen on $p: $!\n";
  $sel->add($l); $lsel->add($l); push @listeners, $l;
}
open(my $log, '>>', "$dir/conn.jsonl") or die "detector: cannot open conn.jsonl: $!\n";
open(my $plog, '>>', "$dir/probe.jsonl") or die "detector: cannot open probe.jsonl: $!\n";
for my $h ($log, $plog) { select((select($h), $| = 1)[0]); }
my %label;               # nonce => label (loaded once; the file is 0600 in a 0700 host directory the components never mount)
if (open(my $th, '<', "$dir/probe-tokens")) {
  while (my $line = <$th>) { $label{$1} = $2 if $line =~ /^([0-9a-f]{32,128}) ([A-Za-z0-9_.\/-]{1,64})$/ }
  close $th;
}
my %used;
sub marker { my ($name, $extra) = @_;
  sysopen(my $h, "$dir/$name.marker", O_WRONLY|O_CREAT|O_EXCL, 0600) or return;
  print $h '{"marker":"' . $name . '","ts":"' . ts() . '"' . ($extra // '') . "}\n"; close $h; }

my %pend;                # fileno => { c, sa, sp, da, dp, t, end, in, first }
sub finish { my $k = shift; my $p = delete $pend{$k}; $sel->remove($p->{c}); close $p->{c};
  my ($out, $tail) = ($log, '');
  if ($p->{first} =~ /\AAC8PROBE:([0-9a-f]{32,128})\n/ && exists $label{$1} && !$used{$1}++) { ($out, $tail) = ($plog, ',"probe":"' . $label{$1} . '"'); }
  print $out '{"ts":"' . ts($p->{t}) . '","srcAddr":"' . $p->{sa} . '","srcPort":' . $p->{sp} . ',"dstAddr":"' . $p->{da} . '","dstPort":' . $p->{dp}
    . ',"bytesOut":0,"bytesIn":' . $p->{in} . $tail . "}\n"; }
# one pass: accept everything queued, read what is readable, log what is finished (EOF or HOLD elapsed)
sub step { my $timeout = shift;
  my @r = $sel->can_read($timeout);
  for my $h (@r) {
    if (grep { $_ == $h } @listeners) {
      for (1 .. 512) {
        my $c = $h->accept or last;
        $c->blocking(0); my $t = time;
        $pend{fileno $c} = { c => $c, sa => $c->peerhost, sp => $c->peerport, da => $c->sockhost, dp => $c->sockport, t => $t, end => $t + $HOLD, in => 0, first => '' };
        $sel->add($c);
      }
    } else {
      my $p = $pend{fileno $h} or next;
      my $n = sysread($h, my $buf, 65536);
      if (defined $n && $n > 0) { $p->{in} += $n; $p->{first} .= $buf if length($p->{first}) < 256; }
      elsif (defined $n || !($!{EAGAIN} || $!{EWOULDBLOCK})) { finish(fileno $h); }   # EOF or a hard error
    }
  }
  my $now = time;
  for my $k (keys %pend) { finish($k) if $pend{$k} && $pend{$k}{end} <= $now; } }
# the accept queues of the listening sockets, from /proc/net/tcp (state 0A: rx_queue is the number of queued, un-accepted connections)
sub queued { my %want = map { $_ => 1 } @ports; my $n = 0;
  open(my $h, '<', '/proc/net/tcp') or return 0;
  while (<$h>) { my @f = split; next unless @f > 4 && $f[3] eq '0A'; my ($pt) = $f[1] =~ /:([0-9A-F]+)$/ or next;
    next unless $want{hex $pt}; my ($rx) = $f[4] =~ /:([0-9A-F]+)$/; $n += hex($rx // '0'); }
  close $h; $n }
# drain until nothing is queued and nothing is being read; returns the residue (0 on a clean drain)
sub drain { my $end = time + $DRAIN_MAX;
  while (time < $end) { step(0.02); last unless %pend || queued() || $lsel->can_read(0); }
  step(0); return scalar(keys %pend) + queued(); }
{ open(my $r, '>', "$dir/ready") or die "detector: $!\n"; print $r "$$\n"; close $r; }
while (1) {
  if ($want_start) { marker('start'); $want_start = 0; }
  if ($want_flush) { my $res = drain(); marker('flush', ',"residue":' . $res); $want_flush = 0; }
  if ($want_term) { my $res = drain(); marker('final', ',"residue":' . $res); exit 0; }
  step(0.1);
}
