#!/usr/bin/perl
# netprobe.pl -- the AC-8 acceptance probe (egress-spec3 §8.1 P-a, P-b, P-c), run from each component container on the
# scenario network. It is Perl because the toolchain image has no Python. One `CHECK <id> PASS|FAIL <detail>` line per check,
# then `PROBE-DONE checks=N fail=M`; the verdict is computed by ac8/assertions.py from this text, never from the exit code.
# A timeout on any connect or send counts as a FAILURE of the isolation (something routed the packet).
# Args, `key=value`, from argv AND from stdin (one per line; stdin wins): v4=<ip,..> v6=<ip,..> tcp=<port,..> udp=<port,..>
#   pos=<host:port,..> det=<host:port,..> tokens=<nonce,..> broker=<socket path> ifaces=<name,..> detname=<detector's container name>
#   detip=<its address>. The harness passes everything but broker/ifaces on STDIN (`docker exec -i`): a nonce on argv is readable from
#   /proc/<pid>/cmdline by any same-uid process in the container and sits in `docker events`. `pos` are plain positive-control targets
#   (the store); `det` are the detector's, to each of which the probe first sends `AC8PROBE:<nonce>\n` (`tokens` is parallel to `det`: one
#   single-use nonce per target) so the detector files the connection as a probe record, not a `conn` (see detector.pl).
use strict; use warnings;
use Socket qw(:all); use POSIX qw(:errno_h); use Fcntl; use Time::HiRes qw(time); use File::Find;
$| = 1;
my %a = map { /^([^=]+)=(.*)$/s ? ($1, $2) : () } @ARGV;
unless (-t STDIN) { while (my $l = <STDIN>) { chomp $l; $a{$1} = $2 if $l =~ /^([^=]+)=(.*)$/s } }
my ($n, $fails) = (0, 0);
sub check { my ($id, $ok, $d) = @_; $n++; $fails++ unless $ok; print "CHECK $id ", ($ok ? 'PASS' : 'FAIL'), " $d\n"; }
sub list { grep { length } split /,/, ($a{$_[0]} // '') }
sub slurp { my $f = shift; open(my $h, '<', $f) or return undef; local $/; my $c = <$h>; close $h; $c }
my %en = map { my $v = eval { no strict 'refs'; &{"POSIX::$_"}() }; defined $v ? ($v => $_) : () }
  qw(ENETUNREACH EHOSTUNREACH EADDRNOTAVAIL ECONNREFUSED EACCES EPERM EAFNOSUPPORT EINPROGRESS ETIMEDOUT EINVAL EROFS);
sub ename { $en{$_[0]} // "errno$_[0]" }
my %UNREACH = map { $_ => 1 } qw(ENETUNREACH EHOSTUNREACH EADDRNOTAVAIL);

# A1(b): no default route, IPv4 or IPv6 (an IPv6 ::/0 entry carrying RTF_REJECT 0x0200 is not a route).
my $r4 = slurp('/proc/net/route') // ''; my @def4 = grep { my @f = split; @f > 7 && $f[1] eq '00000000' && $f[7] eq '00000000' } grep { !/^Iface/ } split /\n/, $r4;
check('A1b-v4', defined $r4 && !@def4, @def4 ? "default route: $def4[0]" : 'no 0.0.0.0/0 in /proc/net/route');
my $r6 = slurp('/proc/net/ipv6_route') // ''; my @def6 = grep { my @f = split; @f >= 9 && $f[0] eq ('0' x 32) && $f[1] eq '00' && !(hex($f[8]) & 0x0200) } split /\n/, $r6;
check('A1b-v6', !@def6, @def6 ? "default route: $def6[0]" : 'no non-reject ::/0 in /proc/net/ipv6_route');

sub tcp { my ($fam, $addr_bin, $port, $scope) = @_;
  socket(my $s, $fam, SOCK_STREAM, 0) or return 'socket:' . ename($! + 0);
  fcntl($s, F_SETFL, fcntl($s, F_GETFL, 0) | O_NONBLOCK);
  my $sa = $fam == AF_INET ? pack_sockaddr_in($port, $addr_bin) : pack_sockaddr_in6($port, $addr_bin, $scope // 0, 0);
  return 'CONNECTED' if connect($s, $sa);
  my $e = $! + 0; return ename($e) unless $e == EINPROGRESS;
  my $w = ''; vec($w, fileno($s), 1) = 1;
  return 'TIMEOUT' unless select(undef, $w, undef, 2) > 0;
  my $so = getsockopt($s, SOL_SOCKET, SO_ERROR); my $err = unpack('i', $so);
  return $err ? ename($err) : 'CONNECTED'; }
sub udp { my ($fam, $addr_bin, $port, $scope) = @_;
  socket(my $s, $fam, SOCK_DGRAM, 0) or return 'socket:' . ename($! + 0);
  my $sa = $fam == AF_INET ? pack_sockaddr_in($port, $addr_bin) : pack_sockaddr_in6($port, $addr_bin, $scope // 0, 0);
  return 'SENT' if defined send($s, 'x', 0, $sa);
  return ename($! + 0); }
sub unreach { my $r = shift; return 1 if $UNREACH{$r}; return 1 if $r =~ /^socket:EAFNOSUPPORT$/; 0 }

# interface scope ids for link-local targets (every container interface except lo)
my @ifs = grep { $_ ne 'lo' } list('ifaces'); my %ifidx;
for my $i (@ifs) { my $x = slurp("/sys/class/net/$i/ifindex"); $ifidx{$i} = $x + 0 if defined $x; }
my @tcpports = list('tcp'); my @udpports = list('udp');
my ($nt, $nu) = (0, 0);
# A1(d): every host address (127/8 and ::1 excluded by the caller) x every listening host port, TCP connect and UDP sendto.
for my $ip (list('v4')) {
  my $ab = inet_aton($ip);
  for my $p (@tcpports) { my $r = tcp(AF_INET, $ab, $p); $nt++; check('A1d-tcp', unreach($r), "$ip:$p $r"); }
  for my $p (@udpports) { my $r = udp(AF_INET, $ab, $p); $nu++; check('A1d-udp', unreach($r), "$ip:$p $r"); }
}
for my $ip (list('v6')) {
  my $ab = inet_pton(AF_INET6, $ip);
  my @scopes = $ip =~ /^fe80:/i ? (map { $ifidx{$_} } grep { defined $ifidx{$_} } @ifs) : (0);
  @scopes = (1) unless @scopes;
  for my $sc (@scopes) {
    for my $p (@tcpports) { my $r = tcp(AF_INET6, $ab, $p, $sc); $nt++; check('A1d-tcp', unreach($r), "[$ip]%$sc:$p $r"); }
    for my $p (@udpports) { my $r = udp(AF_INET6, $ab, $p, $sc); $nu++; check('A1d-udp', unreach($r), "[$ip]%$sc:$p $r"); }
  }
}
# the two public targets: 1.1.1.1:443 and [2606:4700:4700::1111]:443, TCP and UDP
{ my $r = tcp(AF_INET, inet_aton('1.1.1.1'), 443); $nt++; check('A1d-tcp', unreach($r), "1.1.1.1:443 $r");
  $r = udp(AF_INET, inet_aton('1.1.1.1'), 443); $nu++; check('A1d-udp', unreach($r), "1.1.1.1:443 $r");
  my $v6 = inet_pton(AF_INET6, '2606:4700:4700::1111');
  $r = tcp(AF_INET6, $v6, 443, 0); $nt++; check('A1d-tcp', unreach($r), "[2606:4700:4700::1111]:443 $r");
  $r = udp(AF_INET6, $v6, 443, 0); $nu++; check('A1d-udp', unreach($r), "[2606:4700:4700::1111]:443 $r"); }
print "TARGETS tcp=$nt udp=$nu\n";
# positive control (P-a): the same probe code must see the store's and the detector's ports accepting.
for my $t (list('pos')) { my ($h, $p) = $t =~ /^(.*):(\d+)$/ or next;
  my $ip = inet_aton($h); my $r = $ip ? tcp(AF_INET, $ip, $p) : 'unresolved:' . $h;
  check('posctl-tcp', $r eq 'CONNECTED', "$t $r"); }
my $sent = 0; my @nonce = list('tokens'); my $ti = 0;
for my $t (list('det')) { my ($h, $p) = $t =~ /^(.*):(\d+)$/ or next; my $nonce = $nonce[$ti++] // '';
  my $ip = inet_aton($h); my $ok = 0; my $why = 'unresolved';
  if ($ip && socket(my $s, AF_INET, SOCK_STREAM, 0)) {
    $why = 'connect failed'; local $SIG{ALRM} = sub { die "timeout\n" };
    eval { alarm 3; if (connect($s, pack_sockaddr_in($p, $ip))) { syswrite($s, "AC8PROBE:$nonce\n"); $ok = 1; $sent++; $why = 'CONNECTED'; } alarm 0; 1 } or do { $why = 'TIMEOUT'; };
    close $s; }
  check('posctl-det', $ok, "$t $why"); }
print "PROBE-SENT det=$sent\n";
# P-b / A2: no DNS out. `getent ahosts example.com` must exit 2 ("not found") within 5 s; a missing getent (exec fails: 127), a signal or a
# timeout is NOT a pass. Its two positive controls: `getent hosts localhost` succeeds (the tool and nsswitch work), and, when the harness
# names its detector (detname/detip), Docker's embedded DNS (127.0.0.11) resolves that container name to the detector's address while
# example.com fails, so the resolver being cut off from the outside is shown with a resolver that otherwise works.
sub getent { my @args = @_;
  pipe(my $rd, my $wr) or return (-1, '', 0);
  my $pid = fork(); return (-1, '', 0) unless defined $pid;
  if (!$pid) { close $rd; open(STDOUT, '>&', $wr); open(STDERR, '>', '/dev/null'); exec('/usr/bin/getent', @args) or exit 127; }
  close $wr; my $t0 = time; my $out = ''; local $SIG{ALRM} = sub { kill 9, $pid };
  alarm 5; while (my $l = <$rd>) { $out .= $l } waitpid($pid, 0); my $rc = $?; alarm 0; close $rd;
  return ($rc, $out, time - $t0); }
{ my ($rc, undef, $dt) = getent('ahosts', 'example.com');
  check('A2-ahosts', $rc >= 0 && !($rc & 127) && ($rc >> 8) == 2 && $dt < 5, sprintf('getent ahosts example.com status=%d signal=%d in %.1fs (want status 2, not found)', $rc >> 8, $rc & 127, $dt)); }
{ my ($rc, $out) = getent('hosts', 'localhost');
  check('A2-localhost', $rc == 0 && $out =~ /\b(127\.0\.0\.1|::1)\b/, sprintf('getent hosts localhost status=%d %s', $rc >> 8, ($out =~ /^(\S+)/ ? $1 : 'no answer'))); }
if (length($a{detname} // '') && length($a{detip} // '')) {
  my ($rc, $out) = getent('hosts', $a{detname});
  check('A2-embedded', $rc == 0 && $out =~ /^\Q$a{detip}\E\s/, sprintf('getent hosts %s status=%d -> %s', $a{detname}, $rc >> 8, ($out =~ /^(\S+)/ ? $1 : 'no answer'))); }
# P-c / A3: the broker's socket is the only host Unix socket; a connect() to it succeeds; AF_VSOCK fails.
my $bsock = $a{broker} // '';
{ my @socks; my %skip = map { $_ => 1 } qw(/proc /sys /dev);
  local $SIG{__WARN__} = sub { };
  find({ no_chdir => 1, wanted => sub { if ($skip{$File::Find::name}) { $File::Find::prune = 1; return } push @socks, $File::Find::name if -S $File::Find::name }, follow => 0 }, '/');
  my @other = grep { $_ ne $bsock } @socks;
  check('A3-files', !@other, @other ? "sockets besides the broker's: @other" : 'no socket file but the broker\'s');
  check('A3-broker-present', (grep { $_ eq $bsock } @socks) ? 1 : 0, $bsock);
  my @named = grep { -e $_ } ('/var/run/docker.sock', '/run/docker.sock', '/run/WSL', '/tmp/.X11-unix/X0', '/mnt/wslg/PulseServer',
    '/run/dbus/system_bus_socket', '/run/systemd/resolve/io.systemd.Resolve');
  check('A3-named', !@named, @named ? "present: @named" : 'no named host socket');
  my $u = slurp('/proc/net/unix') // ''; my @un = grep { !/^Num/ && /\@/ } split /\n/, $u;
  check('A3-abstract', !@un, @un ? "abstract: $un[0]" : 'no abstract socket'); }
{ socket(my $s, AF_UNIX, SOCK_STREAM, 0); my $ok = connect($s, pack_sockaddr_un($bsock)); my $e = $! + 0; close $s;
  check('A3-connect', $ok ? 1 : 0, $ok ? "connect($bsock) ok" : "connect($bsock): " . ename($e)); }
# the broker directory is mounted read-only: creating a file beside the socket fails with EROFS (an unlink or a bind would too)
{ (my $dir = $bsock) =~ s{/[^/]*$}{}; my $f = "$dir/ac8-write-probe-$$";
  my $ok = sysopen(my $h, $f, O_WRONLY | O_CREAT | O_EXCL, 0600); my $e = $! + 0;
  if ($ok) { close $h; unlink $f; }
  check('A3-readonly', !$ok && $e == EROFS, $ok ? "created $f: the broker directory is writable" : "create beside the socket: " . ename($e)); }
{ my $ok = socket(my $s, 40, SOCK_STREAM, 0); my $e = $! + 0;
  check('A7', !$ok && ($e == EPERM || $e == EAFNOSUPPORT || $e == EACCES), $ok ? 'AF_VSOCK socket created' : 'AF_VSOCK ' . ename($e)); }
print "PROBE-DONE checks=$n fail=$fails\n";
exit($fails ? 1 : 0);
