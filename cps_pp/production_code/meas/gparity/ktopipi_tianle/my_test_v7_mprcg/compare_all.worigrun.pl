#!/usr/bin/perl
import sys;

$ARGC = scalar(@ARGV);

if($ARGC != 1){
    print "Must provide configuration\n";
    sys.exit(0);
}
$conf = $ARGV[0];


$ckroot = ".";
$dqroot = "../full/compare_to_symmpi";

%partners = ( "111"=>"_1_1_1",
	      "_111"=>"1_1_1",
	      "1_11"=>"_11_1",
	      "11_1"=>"_1_11",
	      "_1_1_1"=>"111",
	      "1_1_1"=>"_111",   
	      "_11_1"=>"1_11",
	      "_1_11"=>"11_1" );

@files = glob("$dqroot/traj_${conf}*");

foreach $f (@files){
    if($f=~m/\.dat$/ || $f=~m/\.hexfloat/ || $f=~m/src2s/){
	next;
    }

    $dqfile = $f;
    $f=~m/\/([^\/]+)$/;
    $ckfile = "$ckroot/$1";

    if($ckfile=~m/(.*)_mom([\d_]+?)_mom([\d_]+?)_symm/){
	$pre = $1;
	$psrc1 = $2;
	$psnk1 = $3;
	$psrc2 = $partners{$psrc1};
	$psnk2 = $partners{$psnk1};
	$ckfile = "${pre}_p1src${psrc1}_p2src${psrc2}_p1snk${psnk1}_p2snk${psnk2}_symm";
    }

#traj_0_FigureR_sep2_mom11_1_mom_111.hexfloat
    #"traj_0_FigureR_sep2_p1src_1_11_p2src1_1_1_p1snk13_1_p2snk_1_11_symm

    if(!(-e $ckfile)){
	print "$ckfile does not exist\n";
    }else{
	print "$ckfile $dqfile\n";
	
	$code = system("./get_diff_output.pl $ckfile $dqfile");
	$code = $code >> 8; #remove system exit status
	print "Exit code $code\n";
	if($code != 0){
	    print "Failure detected $ckfile $dqfile\n";
	    sys.exit(-1);
	}
    }
}
print "DONE";
