#!/usr/bin/perl
import sys;

$ARGC = scalar(@ARGV);

if($ARGC < 2){
    print "Need files";
    sys.exit(0);
}
$file1 = $ARGV[0];
$file2 = $ARGV[1];

#print "Files $file1 $file2\n";
#print "Tolerance is $tol\n";

open(F1,$file1);
open(F2,$file2);

@f1 = <F1>;
@f2 = <F2>;

close(F1);
close(F2);

if(scalar(@f1) != scalar(@f2)){
    print "Number of lines is different!\n";
    sys.exit(-1);
}
for($i=0;$i<scalar(@f1);$i++){
    $line1 = $f1[$i];
    $line2 = $f2[$i];

    $tmp = $line1;
    @idx1 = ();
    while($tmp=~m/^(\d+)\s+(.*)/){
	#print "$tmp -> ";
	push(@idx1,$1);
	$tmp = $2;
	#print "($1) + ($2)\n";	
    }
    $vals1 = $tmp;

    $tmp = $line2;
    @idx2 = ();
    while($tmp=~m/^(\d+)\s+(.*)/){
	push(@idx2,$1);
	$tmp = $2;	
    }
    $vals2 = $tmp;
    
    $nidx1 = scalar(@idx1);
    $nidx2 = scalar(@idx2);

    if($nidx1 != $nidx2){
	print "Number of indices on line $i don't match: $nidx1 $nidx2\n";
	print "Lines are:\n${line1}\n${line2}\n";
	sys.exit(-1);
    }
    #print "Line $i matched number of indices $nidx1\n";

    for($j=0;$j<$nidx1;$j++){
	if($idx1[$j] != $idx2[$j]){
	    print "Index match fail: $idx1[$j] $idx2[$j]";
	    sys.exit(-1);
	}
    }
    #print "Line $i matched indices\n";

    $tmp = $vals1;
    @data1 = ();
    while($tmp=~m/^([\d\.e\+\-]+)\s*(.*)/){
	#print "$tmp -> ";
	push(@data1,$1);
	$tmp = $2;
	#print "($1) + ($2)\n";
    }

    $tmp = $vals2;
    @data2 = ();
    while($tmp=~m/^([\d\.e\+\-]+)\s*(.*)/){
	push(@data2,$1);
	$tmp = $2;
    }
    
    $ndata1 = scalar(@data1);
    $ndata2 = scalar(@data2);
    if($nidx1 != $nidx2){
	print "Number of data on line $i don't match: $ndata1 $ndata2\n";
	print "Lines are:\n${line1}\n${line2}\n";
	sys.exit(-1);
    }
    #print "Line $i matched number of data $ndata1\n";

    $fail = 0;
    for($j=0;$j<$ndata1;$j++){
	if($data1[$j] == 0.0 && $data2[$j] == 0.0){
	    next;
	}
	if( ($data1[$j] == 0.0 && $data2[$j] != 0.0) || ($data2[$j] == 0.0 && $data1[$j] != 0.0) ){
	    #print "WARNING one datum is zero: $j $data1[$j] $data2[$j] in files $file1 $file2\n";
	    #print "Line1: $line1";
	    #print "Line2: $line2";
	    next;
	}

	$reldiff = 2.0 * abs( ($data1[$j] - $data2[$j])/($data1[$j] + $data2[$j]) );

#	print "$j $data1[$j] $data2[$j]  reldiff $reldiff\n";
	printf("%d %e %e reldiff %e\n", $j, $data1[$j], $data2[$j], $reldiff);
    }
}

