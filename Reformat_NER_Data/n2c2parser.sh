# loop through two directories simultaneously and run parser on files with same name in both directories

# directory containing files to be parsed
# dir1="training_20180910/ann/"
# dir2="training_20180910/txt/"
dir1="test/ann/"
dir2="test/txt/"
dir1="sr/Reformat_SR_Data/Barilla_SR_Example/Barilla_Training/txt"
dir2="sr/Reformat_SR_Data/Barilla_SR_Example/Barilla_Training/ann"


# directory to store parsed files
# out_dir="training_20180910/parsed/"
# out_dir="test/parsed/"
out_dir="sr/Reformat_SR_Data/Barilla_SR_Example/Barilla_Training/parsed/"

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
