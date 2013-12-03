# meant to be run from (build directory)/tests
#!/bin/tcsh
set summary = compile_summary.log
set output = compile.log
set err_log = error_summary.log
rm -f $summary
foreach dir ( `find . -type d`) 
	echo $dir 
	pushd $dir
	make -f Makefile_regression  clean all >& $output
	popd
	echo '====== '$dir" =======" >> $summary
	tail -10 $dir/$output >> $summary
end
rm -f $err_log
foreach err ( `grep -l Error */$output`) 
	echo '====== '$err" =======" >> $err_log
	tail -10 $err >> $err_log
end

