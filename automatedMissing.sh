targetDir='/10077/data/Micrographs_part1/'
dirName='./empiar10077/Micrographs_part1/'
resultLog='./reResult.txt'

# 1. find the files not downloaded. 
> $resultLog # empty the reResult.txt file.
while read line; do 
    # echo $line;
    # echo "$dirName$line";
    if [ -f "$dirName$line" ] ; then
        # echo "file $dirName$line exist"
        :
    else
        # echo "file $dirName$line does not exist"
        echo "$line" >> $resultLog
    fi
done < tempArg.txt;

# 2. remove partial downloaded files.
# rm "${dirName}*.partial"

# 3. download the missing files
cat $resultLog | while read p; do
    echo "host : ~/.aspera/connect/etc/asperaweb_id_dsa.openssh emp_ext3@fasp.ebi.ac.uk:${targetDir}${p}"
    ~/.aspera/connect/bin/ascp -QT -P33001 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh "emp_ext3@fasp.ebi.ac.uk:${targetDir}${p}" "${dirName}${p}"
done

echo "All of the jobs are finished!"