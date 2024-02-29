#!/usr/bin/perl
import sys;

$ARGC = scalar(@ARGV);

if($ARGC != 1){
    print "Must provide configuration\n";
    sys.exit(0);
}
$conf = $ARGV[0];


$ckroot = "./results/";
$dqroot = "/ccs/home/tianle/Record/latticd_qcd/run_10_15_18";


#@files = glob("$dqroot/traj_${conf}*");
@files = glob("$ckroot/traj_${conf}*");

foreach $f (@files){
    if($f=~m/\.dat$/ || $f=~m/\.hexfloat/ ){ #|| $f=~m/symmpi/
	next;
    }

#    $dqfile = $f;
#    $f=~m/\/([^\/]+)$/;
#    $ckfile = "$ckroot/$1";
 
    $ckfile = $f;
    $f=~m/\/([^\/]+)$/;
    $dqfile = "$dqroot/$1";

  
    print "$ckfile $dqfile\n";
    if(!(-e $dqfile)){
	print "$dqfile not present in comparison data\n";
	next;
    }

    $code = system("./get_diff_output.pl $ckfile $dqfile");
    $code = $code >> 8; #remove system exit status
    print "Exit code $code\n";
    if($code != 0){
	print "Failure detected $ckfile $dqfile\n";
	sys.exit(-1);
    }
}
print "DONE";
