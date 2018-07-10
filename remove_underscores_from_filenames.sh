for i in $(ls | egrep 'murmur__[0-9]*_')
do
    mv "$i" "`echo $i | sed 's/___/_/'`"
done


