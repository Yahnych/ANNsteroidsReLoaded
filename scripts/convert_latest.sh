#!/bin/bash
# convert_latest.sh --- convert the latest 
# directories to png and remove the rest
usage(){
	echo "Usage: $0 <last_updated> <exclude>"
}

PREVIOUS=$1
EXCLUDE=$2

ls -d ep* | grep -v $EXCLUDE > current_hist.txt
diff current_hist.txt $PREVIOUS > diff_.txt

cat diff_.txt | sed 's/<//' | sed 's/ //' | grep ep > todo.txt

for ep in $(cat todo.txt); do
	echo "==== $ep ===="
 	./scripts/to_png.sh $ep
	rm $ep/*pgm
	rm $ep/rsz/*pgm
	#python Infographics/Gif_Creator.py $ep/pngs $ep $ep
	#python Infographics/Gif_Creator.py $ep/pngs/rsz $ep "$ep-rsz"

done 	
