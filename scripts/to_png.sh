DIR=$1

mkdir $DIR/pngs
PNGDIR=$DIR/pngs

CURSOR_UP_ONE="\x1b[1A"
ERASE_LINE="\x1b[2K"

BUFFER_LINE=$CURSOR_UP_ONE+$ERASE_LINE

for f in $DIR/*pgm; do
	PNGSTR=$(echo $f | sed 's/.pgm/.png/')
echo "$f,  $PNGSTR"
magick $f $PNGSTR
done

mv $DIR/*png $PNGDIR
mkdir $PNGDIR/rsz
RSZDIR=$PNGDIR/rsz

for rsz_f in $DIR/rsz/*pgm; do
	RSZ_STR=$(echo $rsz_f | sed 's/.pgm/.png/')
echo "$rsz_f, $RSZ_STR"
magick $rsz_f $RSZ_STR 
done

mv $DIR/rsz/*png $RSZDIR

#python gif_it.py $RSZDIR/*png $RSZDIR $DIR
