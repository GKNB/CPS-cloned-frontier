$l_Ls=shift;
$s_Ls=shift;
$fil=shift;
#print "opening $fil\n";

$tmpfil="/tmp/cnt.$$";
#print "grep True $fil | grep Inv > $tmpfil";
system "grep True $fil | grep Inv > $tmpfil";
open(P, $tmpfil);

$_ = `wc $tmpfil`;
split;
$n=$_[0];

$ncycle = int($n / 3);

print "ncycle = $ncycle\n";




$cnt_s = 0;
$cnt_l = 0;

$_=<P>;

#print; exit(1);
if(/iter = (\d+)/){
    $cnt_l += $1;
    printf( "L %3d\n",$1);
}else{die;};


for($i=0;$i<$ncycle;++$i){
$_=<P>;
if(/iter = (\d+)/){
    $cnt_s += $1;
    printf( "S %3d\n",$1);
}else{die;};

$_=<P>;
if(/iter = (\d+)/){
    $cnt_s += $1;
    printf( "S %3d\n",$1);
}else{die "$_";};

$_=<P>;
if(/iter = (\d+)/){
    $cnt_l += $1;
    printf( "L %3d\n",$1);
}else{die;};
}

$_=<P>;
if(/iter = (\d+)/){
    $cnt_l += $1;
    printf( "L %3d\n",$1);
}else{die;};

close P;

printf("iterations of Ls=%2d : %4d\n",$l_Ls,$cnt_l);
printf("iterations of Ls=%2d : %4d\n",$s_Ls,$cnt_s);

$scaled_cnt = $cnt_l + $cnt_s*($s_Ls/$l_Ls);

printf("scaled iterations : %4d\n", $scaled_cnt);

