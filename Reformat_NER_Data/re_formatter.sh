# loop through two directories simultaneously and run parser on files with same name in both directories

# directory containing files to be parsed
dir1="./data/ann/"
dir2="./data/txt/"

# directory to store parsed files
out_dir="./data/re_json/"

parse() {
    file1=$1
    ficd ..le2=$dir2$(basename -s .ann $file1).txt
    out_file=$out_dir$(basename -s .ann $file1).json
    echo $file1
    echo $file2
    echo $out_file
    python3 re_formatter.py $file1 $file2 $out_file
}

# loop through files in both directories
for file1 in $dir1*.ann; do
    parse $file1 &
done