import os

main_dir = "J:/Documents/Datahound/DirectorySearch"

dirNames = ['SearchFolder.{}'.format(i) for i in range(1000)]
for dirname in dirNames:
    dirCreator = os.path.join(main_dir, dirname)
    os.mkdir(dirCreator)

