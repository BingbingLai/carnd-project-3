model28 is good; with the following dirs
- dir: local-trained-data-curves-new, images: 30054
- dir: local-trained-data-opposite-1, images: 16734
- dir: local-trained-data-original-direction, images: 33114
- dir: local-trained-data-drive-to-center, images: 10026


model29 is terrible
- dir: data (from udacity) terrible combine with local trained data


model30 is not bad but it falls off on whitelines
- may need to teach it to steer away from white line

model31 is still with the curvenews

model32 is from last night -- it's no good

model33 will be the one that's only using the new data


# TODO
- [DONE] maybe for next one try to get rid of the curve_new data and try again?
- [DONE] maybe also flip the images
- think the following training images should be good:
  - [DONE] 2/3 laps of normal driving in the middle
  - [DONE] 2/3 laps of opposite driving in the middle
  - [DONE] 2/3 laps of driving to the center from the side (this is probably good enough)
