# CropImage: Cropping an image from a ScanSnap-scanned image.

A raw scanned image of ScanSnap with SANE includes surrounding margin.
To make e-books by scanning pages of real books (自炊), the margins should be detected and removed automatically.
A trial to do that using OpenCV.

## Developping environments

- OS: FreeBSD 15.0
- OpenCV 4.12 with GTK3
- Image scanner: Richo ScanSnap iX1600
- SANE 1.4 (ports: graphics/sane-backends)
- Simple Scan (ports: graphics/simple-scan)

## How to compile (personal memorandum)

```
# mkdir build; cd build
# cmake ..; make
```

## Basic concept of margin detection

A ScanSnap-scanned image seems to have a slightly greenish background.
To detect a margin area, the difference between the Green-channel and the Red-channel of the scanned image is calculated and enhanced by OpenCV.
This works almost well when the scanned image is BW-like, but when the image is colorful (especially greenish), detection may fail.
