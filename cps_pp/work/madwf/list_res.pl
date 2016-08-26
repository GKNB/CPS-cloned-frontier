$fil=shift;
#print "opening $fil\n";
open(I,"$fil");

$irestart=0;
$istep=-1;
open O,"> res-$irestart";
$old=-100;
while(<I>){
    if(/\|res\[(\d+)\]\|\^2 = (\S+)/){
	if($1 <$old){
	    close O;
	    $istep++;
	    if($istep%3==0){
		$istep=0;
		$irestart++;
	    }
	    @str=('A-sPV','B-s','C-lPV');
	    $fname=sprintf("res-%02d-%s", $irestart,$str[$istep]);
	    open O,"> $fname";
	}
	print O $1,"  ", $2,"\n";
	$old=$1;
    }
}

close O;
close I;

rename $fname,sprintf("res-%02d-lFin",$irestart);
