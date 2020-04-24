for f in *.drawio; do
    drawio --export $f -q 100 -s 3 --output "${f%.drawio}.jpeg"
done
mv *.jpeg img

