import panorama.panorama as panorama

from PIL import Image

uttower_left = Image.open('data/uttower_left.JPG')
uttower_right = Image.open('data/uttower_right.JPG')
# hill_1 = Image.open('data/hill/1.JPG')
# hill_2 = Image.open('data/hill/2.JPG')
# hill_3 = Image.open('data/hill/3.JPG')
# ledge_1 = Image.open('data/ledge/1.JPG')
# ledge_2 = Image.open('data/ledge/2.JPG')
# ledge_3 = Image.open('data/ledge/3.JPG')
pier_1 = Image.open('data/pier/1.JPG')
pier_2 = Image.open('data/pier/2.JPG')
pier_3 = Image.open('data/pier/3.JPG')

uttower = panorama.generate([uttower_left, uttower_right])
uttower.save('output/uttower.png')

# hill = panorama.generate([hill_1, hill_2, hill_3])
# hill.save('output/hill.png')

# ledge = panorama.generate([ledge_1, ledge_2, ledge_3])
# ledge.save('output/ledge.png')

pier = panorama.generate([pier_1, pier_2, pier_3])
pier.save('output/pier.png')
