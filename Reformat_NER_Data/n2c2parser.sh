# loop through two directories simultaneously and run parser on files with same name in both directories

# directory containing files to be parsed
dir1="training_20180910/ann/"
dir2="training_20180910/txt/"

# directory to store parsed files
out_dir="training_20180910/parsed/"

parse() {
    file1=$1
    file2=$dir2$(basename -s .ann $file1).txt
    out_file=$out_dir$(basename -s .ann $file1).tsv
    echo $file1
    echo $file2
    echo $out_file
    python3 standoff_tsv.py $file1 $file2 $out_file
}

# loop through files in both directories
for file1 in $dir1*.ann; do
    parse $file1 &
done
