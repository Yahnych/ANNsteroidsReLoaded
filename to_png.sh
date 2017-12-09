mkdir pngs

for f in *pgm; do
substr=$(echo ${f:0:-4}.png)
magick $f pngs/$substr.png
done
