$old=-100;
while(<>){

    if(/\|res\[(\d+)\]\|\^2 = (\S+)/){
	print $1,"  ", $2,"\n";
	if($1 <$old){print "\n";}
	$old=$1;
    }
}
