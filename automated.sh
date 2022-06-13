targetDir='/10077/data/Micrographs_part1/'
dirName='./empiar10077/Micrographs_part1/'

# Download the index.html
wget "http://ftp.ebi.ac.uk/empiar/world_availability${targetDir}" -o temp.html
# Parse the file. Get the file list.
grep -oi "href=\".*\"" index.html | grep -oi \".*\" | grep -v "../" > tempArg.txt
sed -i 's/"//g' tempArg.txt
sed -i 's/%20/ /g' tempArg.txt
# Get the file names in the tempArg2.txt
rm temp.html index.html
cat tempArg.txt | while read p; do
    echo "host : ~/.aspera/connect/etc/asperaweb_id_dsa.openssh emp_ext3@fasp.ebi.ac.uk:${targetDir}${p}"
    ~/.aspera/connect/bin/ascp -QT -P33001 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh "emp_ext3@fasp.ebi.ac.uk:${targetDir}${p}" "${dirName}${p}"
done
# mv "tempArg.txt" "${dirName}/catalogue.txt"
echo "All of the jobs are finished!"
# wget http://ftp.ebi.ac.uk/empiar/world_availability/10077/data/Micrographs_part1/sb1_210512%20pos%2010%201-1_1.mrc
# sb1_210512%20pos%2010%203-2_1.mrc
# sb1_210512 pos 10 3-2
# ascp -QT -P33001 -i "~/.aspera/connect/etc/asperaweb_id_dsa.openssh emp_ext3@fasp.ebi.ac.uk:/10077/data/Micrographs_part1/sb1_210512%20pos%2010%203-2_1.mrc" "./empiar10077/Micrographs_part1/sb1_210512%20pos%2010%203-2_1.mrc"