#!/usr/bin/perl
# probe.pl — S08 escape-axis probe (egress spec §7.2, axes A1–A7). Launched by test_sandbox.sh
# THROUGH the runner's launch path (sbx_launch), exactly as a tool is: `env -i <pass-list>
# /usr/bin/perl /src/probe.pl key=value...`. Perl is used because the toolchain image has no
# Python, and Perl's Socket/syscall give raw errnos, AF_VSOCK and memfd_create.
#
# Output, one per line:  CHECK <id> PASS|FAIL <detail>   ENV <key>   JAVA_OPTS <value>
#   ENVV <key=value>   CHILDENV <bash|sh> <key>   MOUNT <mountpoint> <options> <fstype>   PROBE-DONE checks=N fail=M
# A timeout on any connect is a FAIL (§7.2: it means something routed the packet).
#
# Args (all optional except axes): axes=A1,A2,... hostns=<net ns link> v4=<ip,...> v6=<ip,...>
#   ports=<p,...> pe=<path of the PE staged in /src> pebinfmt=0|1 interp=<path> uid=<n>
#   sentinels=<path:path> hostpid=<pid> marker=<string>
use strict; use warnings;
use Socket qw(:all);
use POSIX qw(:errno_h _exit);
use Fcntl;
use Time::HiRes qw(time);

$| = 1;
my %a = map { /^([^=]+)=(.*)$/s ? ($1, $2) : () } @ARGV;
my %axes = map { $_ => 1 } split /,/, ($a{axes} // 'A1,A2,A3,A4,A5,A6,A7');
my ($n, $fails) = (0, 0);
sub check { my ($id, $ok, $d) = @_; $n++; $fails++ unless $ok; print "CHECK $id ", ($ok ? 'PASS' : 'FAIL'), " $d\n"; }
my %en; for my $nm (qw(ENETUNREACH EHOSTUNREACH EADDRNOTAVAIL ECONNREFUSED EACCES ENOENT ENOEXEC EPERM EAFNOSUPPORT EROFS EINPROGRESS ETIMEDOUT)) {
  no strict 'refs'; my $v = eval { &{"POSIX::$nm"}() }; $en{$v} = $nm if defined $v; }
sub ename { my $e = shift; $en{$e} // "errno$e" }
sub slurp { my $f = shift; open(my $h, '<', $f) or return undef; local $/; my $c = <$h>; close $h; $c }
sub list { grep { length } split /,/, ($a{$_[0]} // '') }

sub saddr { my ($fam, $ip, $port) = @_;
  $fam == AF_INET ? pack_sockaddr_in($port, inet_aton($ip)) : pack_sockaddr_in6($port, inet_pton(AF_INET6, $ip)); }
# TCP connect with a 2 s bound: returns CONNECTED | TIMEOUT | errno name.
sub tcp { my ($fam, $ip, $port) = @_;
  socket(my $s, $fam, SOCK_STREAM, 0) or return 'socket:' . ename($! + 0);
  fcntl($s, F_SETFL, fcntl($s, F_GETFL, 0) | O_NONBLOCK);
  return 'CONNECTED' if connect($s, saddr($fam, $ip, $port));
  my $e = $! + 0;
  return ename($e) unless $e == EINPROGRESS;
  my $w = ''; vec($w, fileno($s), 1) = 1;
  my $r = select(undef, $w, undef, 2);
  return 'TIMEOUT' unless $r;
  my $err = unpack('i', getsockopt($s, SOL_SOCKET, SO_ERROR));
  $err ? ename($err) : 'CONNECTED'; }
sub udp_sendto { my ($fam, $ip, $port) = @_;
  socket(my $s, $fam, SOCK_DGRAM, 0) or return 'socket:' . ename($! + 0);
  defined(send($s, 'x', 0, saddr($fam, $ip, $port))) ? 'SENT' : ename($! + 0); }
# Loopback UDP (A1e): connect, send 1 byte, recv with a 2 s timeout.
sub udp_loop { my ($fam, $ip, $port) = @_;
  socket(my $s, $fam, SOCK_DGRAM, 0) or return 'socket:' . ename($! + 0);
  connect($s, saddr($fam, $ip, $port)) or return 'connect:' . ename($! + 0);
  defined(send($s, 'x', 0)) or return 'send:' . ename($! + 0);
  my $rd = ''; vec($rd, fileno($s), 1) = 1;
  return 'TIMEOUT' unless select($rd, undef, undef, 2);
  my $buf; defined(recv($s, $buf, 64, 0)) ? 'DATAGRAM' : 'recv:' . ename($! + 0); }
my %unreach = map { $_ => 1 } qw(ENETUNREACH EHOSTUNREACH EADDRNOTAVAIL);

# ---------------------------------------------------------------- A1 IP
if ($axes{A1}) {
  my @dev = grep { /:/ } split /\n/, (slurp('/proc/net/dev') // '');
  my @ifs = map { /^\s*([^:]+):/ ? $1 : () } @dev;
  check('A1a', "@ifs" eq 'lo', "interfaces=[@ifs]");
  my @r4 = grep { !/^Iface/ } split /\n/, (slurp('/proc/net/route') // '');
  my @d4 = grep { (split /\s+/)[1] eq '00000000' } @r4;
  # A ::/0 entry flagged RTF_REJECT (0x0200) is the kernel's ip6_null_entry, which every netns
  # carries (metric ffffffff, dev lo): it rejects, so it is not a route out.
  my @d6 = grep { my @f = split /\s+/; $f[0] eq ('0' x 32) && $f[1] eq '00' && !(hex($f[8]) & 0x0200) } split /\n/, (slurp('/proc/net/ipv6_route') // '');
  check('A1b', !@d4 && !@d6, 'default routes v4=' . scalar(@d4) . ' v6=' . scalar(@d6));
  my $ns = readlink('/proc/self/ns/net') // '?';
  check('A1c', defined $a{hostns} && $ns ne $a{hostns}, "sandbox=$ns host=" . ($a{hostns} // 'unset'));
  my @ports = list('ports');
  my @t = ( [AF_INET, '1.1.1.1', 443], [AF_INET6, '2606:4700:4700::1111', 443] );
  for my $ip (list('ns53')) { push @t, [($ip =~ /:/ ? AF_INET6 : AF_INET), $ip, 53] }
  for my $ip (list('v4')) { push @t, [AF_INET, $ip, $_] for @ports }
  for my $ip (list('v6')) { push @t, [AF_INET6, $ip, $_] for @ports }
  for my $x (@t) {
    my ($fam, $ip, $p) = @$x; my $t0 = time;
    my $r = tcp($fam, $ip, $p); my $dt = time - $t0;
    my $ok = ($unreach{$r} || $r eq 'socket:EAFNOSUPPORT') && $dt < 2.5;
    check('A1d-tcp', $ok, sprintf("%s:%d -> %s (%.2fs)", $ip, $p, $r, $dt));
    $r = udp_sendto($fam, $ip, $p);
    check('A1d-udp', $unreach{$r} || $r eq 'socket:EAFNOSUPPORT', "$ip:$p sendto -> $r");
  }
  for my $x ([AF_INET, '127.0.0.1'], [AF_INET6, '::1']) {
    my ($fam, $ip) = @$x;
    for my $p (@ports) {
      my $r = tcp($fam, $ip, $p);
      check('A1e-tcp', $r eq 'ECONNREFUSED' || $r eq 'ENETUNREACH' || $r eq 'EADDRNOTAVAIL' || $r eq 'socket:EAFNOSUPPORT', "$ip:$p -> $r");
      $r = udp_loop($fam, $ip, $p);
      my $ok = $r =~ /^(send|recv):ECONNREFUSED$/ || $r =~ /^(connect|send):(ENETUNREACH|EADDRNOTAVAIL)$/ || $r eq 'socket:EAFNOSUPPORT';
      check('A1e-udp', $ok, "$ip:$p -> $r");
    }
  }
}

# ---------------------------------------------------------------- A2 DNS
if ($axes{A2}) {
  my $t0 = time; my $rc = system('/usr/bin/timeout 6 /usr/bin/getent ahosts example.com >/dev/null 2>&1') >> 8; my $dt = time - $t0;
  check('A2a', $rc != 0 && $dt < 5, sprintf("getent ahosts example.com rc=%d %.2fs", $rc, $dt));
  $rc = system('/usr/bin/getent hosts localhost >/dev/null 2>&1') >> 8;
  check('A2b', $rc == 0, "getent hosts localhost rc=$rc");
}

# ---------------------------------------------------------------- A3 host Unix sockets
if ($axes{A3}) {
  my $out = `/usr/bin/find / -path /proc -prune -o -path /sys -prune -o -type s -print 2>/dev/null`;
  check('A3a', $out eq '', 'sockets=[' . join(' ', split /\n/, $out) . ']');
  my @u = split /\n/, (slurp('/proc/net/unix') // '');
  check('A3b', @u == 1, '/proc/net/unix lines=' . scalar(@u));
  my $uid = $a{uid} // 1000;
  my @named = ('/run/docker.sock', '/var/run/docker.sock', glob('/run/WSL/*_interop'), '/tmp/.X11-unix/X0',
    '/mnt/wslg/PulseServer', "/run/user/$uid/bus", '/run/dbus/system_bus_socket', '/run/systemd/resolve/io.systemd.Resolve');
  my @present = grep { -e $_ } @named;
  check('A3c', !@present, 'present=[' . join(' ', @present) . '] /run/WSL ' . (-e '/run/WSL' ? 'exists' : 'absent'));
}

# ---------------------------------------------------------------- A4 WSL interop exec
# Run <path> in a child; returns (errno name of the failed execve | "EXITED <rc>", stdout).
sub try_exec { my $path = shift;
  pipe(my $er, my $ew) or die; pipe(my $or, my $ow) or die;
  my $pid = fork() // die;
  if (!$pid) { close $er; close $or; open(STDOUT, '>&', $ow); open(STDERR, '>', '/dev/null');
    { no warnings 'exec'; exec { $path } $path; } syswrite($ew, ($! + 0) . "\n"); _exit(127); }
  close $ew; close $ow;
  my $e = do { local $/; <$er> } // ''; my $o = do { local $/; <$or> } // '';
  waitpid($pid, 0); my $rc = $? >> 8;
  chomp $e; ($e ne '' ? ename($e) : "EXITED $rc", $o); }
if ($axes{A4}) {
  my $pebytes = defined $a{pe} ? slurp($a{pe}) : undef;
  $pebytes = "MZ" unless defined $pebytes && length $pebytes;
  my $want_tmp = ($a{pebinfmt} // 1) ? 'ENOENT' : 'ENOEXEC';
  for my $d ('/src', '/out', '/tmp', '/home/ccg') {
    my $p;
    if ($d eq '/src') { $p = $a{pe} // '/src/whoami.exe'; }
    else { $p = "$d/pe-probe.exe"; open(my $h, '>', $p) or do { check("A4a$d", 0, "cannot stage $p: $!"); next };
      binmode $h; print $h $pebytes; close $h; chmod 0755, $p; }
    my ($r, $o) = try_exec($p);
    my $ok = $o eq '' && ($d eq '/tmp' ? $r eq $want_tmp : $r =~ /^(ENOENT|EACCES|ENOEXEC)$/);
    check("A4a:$d", $ok, "$p -> $r" . ($o ne '' ? " stdout=[$o]" : '') . ($d eq '/tmp' ? " (want $want_tmp)" : ''));
  }
  # (a') memfd_create (x86_64 syscall 319): no mount flag can stop this exec.
  my $fd = syscall(319, my $nm = "pe", 0);
  if ($fd < 0) { check('A4a-memfd', 0, 'memfd_create failed: ' . ename($! + 0)); }
  else { open(my $mh, '>&=', $fd) or die; binmode $mh; print $mh $pebytes; $mh->flush;
    my ($r, $o) = try_exec("/proc/self/fd/$fd");
    check('A4a-memfd', $r eq $want_tmp && $o eq '', "/proc/self/fd/$fd -> $r (want $want_tmp)"); }
  my $interp = $a{interp} // '/init';
  check('A4b', !exists $ENV{WSL_INTEROP} && !-e '/run/WSL' && !-e $interp,
    'WSL_INTEROP=' . (exists $ENV{WSL_INTEROP} ? 'set' : 'unset') . ' /run/WSL=' . (-e '/run/WSL' ? 'present' : 'absent') . " $interp=" . (-e $interp ? 'present' : 'absent'));
  my $made = open(my $ih, '>', $interp); my $why = ename($! + 0);
  if ($made) { close $ih; unlink $interp; }
  check('A4c', !$made, "create $interp -> " . ($made ? 'CREATED' : $why));
}

# ---------------------------------------------------------------- A5 mounted host / Windows paths
if ($axes{A5}) {
  my @bad;
  for my $l (split /\n/, (slurp('/proc/self/mountinfo') // '')) {
    my ($pre, $post) = split / - /, $l, 2;
    my @f = split / /, $pre; my ($fstype) = split / /, $post;
    my $mp = $f[4]; $mp =~ s/\\([0-7]{3})/chr(oct($1))/ge;
    print "MOUNT $mp $f[5] $fstype\n";
    # n2: no mount point at all under /mnt, /usr/lib/wsl or /run/WSL (not even a tmpfs mask).
    push @bad, "$mp($fstype)" if $fstype =~ /^(9p|drvfs|virtiofs)$/ || $mp =~ m{^/(mnt|usr/lib/wsl|run/WSL)(/|$)} || $mp eq '/init';
  }
  check('A5b', !@bad, 'bad=[' . join(' ', @bad) . ']');
  my @seen = grep { -e $_ } split /:/, ($a{sentinels} // '');
  check('A5c', defined $a{sentinels} && !@seen, 'visible=[' . join(' ', @seen) . ']');
  my @pids = grep { /^\d+$/ } map { s{^/proc/}{}r } glob('/proc/[0-9]*');
  my $hp = $a{hostpid} // 0; my $cl = slurp("/proc/$hp/cmdline") // '';
  my $marker = $a{marker} // "\0none";
  check('A5d', index($cl, $marker) < 0 && @pids < 64, 'pids=' . scalar(@pids) . " hostpid $hp " . (index($cl, $marker) >= 0 ? 'VISIBLE' : 'not visible'));
}

# ---------------------------------------------------------------- A6 environment (raw data; the test compares)
if ($axes{A6}) {
  my $env = slurp('/proc/self/environ') // '';
  for my $kv (split /\0/, $env) { my ($k) = split /=/, $kv, 2; print "ENV $k\n"; print "ENVV $kv\n"; }
  print "JAVA_OPTS ", ($ENV{JAVA_OPTS} // ''), "\n";
  for my $sh ('bash', 'sh') {
    my $c = `/bin/$sh -c 'cat /proc/self/environ; :'`;
    for my $kv (split /\0/, $c) { my ($k) = split /=/, $kv, 2; print "CHILDENV $sh $k\n"; }
  }
  check('A6-read', length($env) > 0, 'own environ read');
}

# ---------------------------------------------------------------- A7 AF_VSOCK
if ($axes{A7}) {
  my $ok = socket(my $v, 40, SOCK_STREAM, 0); my $e = ename($! + 0);
  check('A7', !$ok && $e =~ /^(EPERM|EAFNOSUPPORT|EACCES)$/, 'socket(AF_VSOCK) -> ' . ($ok ? 'CREATED' : $e));
}

print "PROBE-DONE checks=$n fail=$fails\n";
exit($fails ? 1 : 0);
