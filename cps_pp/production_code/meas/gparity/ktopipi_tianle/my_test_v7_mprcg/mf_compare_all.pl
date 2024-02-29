#!/usr/bin/perl
import sys;

$ARGC = scalar(@ARGV);

if($ARGC != 1){
    print "Must provide configuration\n";
    sys.exit(0);
}
$conf = $ARGV[0];


$ckroot = ".";
$dqroot = "compare_to_symm";


@files = glob("$ckroot/traj_${conf}*.dat");

foreach $f (@files){
    $ckfile = $f;
    $f=~m/\/([^\/]+)$/;
    $dqfile = "$dqroot/$1";
    
    print "$ckfile $dqfile\n";

    if(!(-e $dqfile)){
	print "$dqfile does not exist!\n";
    }else{
	$code = system("./mf_compare.sh $ckfile $dqfile");
	$code = $code >> 8; #remove system exit status
	# print "Exit code $code\n";
	# if($code != 0){
	# 	print "Failure detected $ckfile $dqfile\n";
	# 	sys.exit(-1);
	# }
    }
}
print "DONE";
