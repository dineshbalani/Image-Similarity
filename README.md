# Image-Similarity
Find similar regions of Long Island by comparing satellite imagery. 
Objectives:

    Implement Locality Sensitive Hashing.
    Implement dimensionality reduction -- learn through doing.
    Gain further experience with Spark.
    Gain further experience with data preprocessing.
    Explore a different modality of data: satellite images.

Data:

The data are high resolution orthorectified satellite images. Orthorectified images satellite pictures which have been corrected for abnormalities due to tilt in photography or geography. Read more here: https://lta.cr.usgs.gov/high_res_ortho

There are two datasets:

    small_sample: hdfs:/data/small_sample http://www3.cs.stonybrook.edu/~has/CSE545/assignments/a2_small_sample.zip -- 5 orthorectified images; small enough for ones own machine and for developing the code
    large_sample: hdfs:/data/large_sample
    Approximately 50 images -- full data set -- the data we will test

Within each sample are additional zip files which contain, among other information, a tiff image file that defines each pixel of the image in terms of red, green, blue, and infrared (i.e. each pixel is represented by 4 integers). Copy one of the zip files and take a look inside to find the tiff file and view it.

Step 1. Read image files into an RDD and divide into 500x500 images (25 points)

    a. Create an rdd of the orthographic zip files in the specified location. Pull the filenames out from the rest of the path (e.g. ‘/data/ortho/small_sample/3677453_2025190.zip’ => ‘3677453_2025190.zip’) and make sure to hang on to it.
    b. Find all the tif files and convert them into arrays (make sure to hang on to the image filename).The following method may be useful: 
    c. Divide the image into 500x500 subimages. Each tif image should be 2500x2500 or 5000x5000 pixels, so you will end up with 25 500x500 images (image breaking the image into a 5x5 grid) or 100 500x500 images.
    d. Produce a final RDD where each record is of the form: (imagename, array).
    Name for each 500x500 image (i.e. matrix) with the file name it was derived along with the cell number of the 500x500 pixel, labeling by rows and then columns (as depicted to the right). For example, if the file was “3677454_2025195.zip” the 500x500 portion corresponding to cell 18 would be labeled “3677454_2025195.zip-18”

Example output:

[('3677454_2025195.zip-0', array([114, 111, 109, 114], dtype=uint8)),

 ('3677454_2025195.zip-1', array([ 54,  53,  57, 117], dtype=uint8)),

 ('3677454_2025195.zip-18', array([ 79,  70,  66, 123], dtype=uint8)),

 ('3677454_2025195.zip-19', array([61, 57, 63, 84], dtype=uint8))]

Step 2. Turn each image into a feature vector

For each 500x500 image (each image should be an RDD record):

    a. Convert the 4 values of each pixel into a single value, intensity = int(rgb_mean * (infrared/100))
	    (i) rgb_mean: the mean value of the red, green, and blue values for the pixel
    	(ii) infrared: (the 4th value in the tuple for the pixel

	E.g. if the pixel was (10, 20, 30, 65), rgb_mean would be 20 and infrared would be 65.
	        Thus, intensity = int(20 * (65/100)) = 13

    b. Reduce the resolution of each image by factor = 10. For example, from 500x500 to 50x50. To do this, take the mean intensity over factor x factor (10x10) sub-matrices. (Note one might build on the method for step 1(c); factor may change for later steps).
    c. Compute the row difference in intensities. Direct intensity is not that useful because of shadows. Instead we focus on change in intensity by computing how it changes from one pixel to the next. Specifically we say intensity[i] = intensity[i+1] - intensity[i] (the numpy function ‘diff’ will do this). Then we turn all values into 3 possible values: Convert all values < -1 to -1, those > 1 to 1, and the rest to 0.
    e.g. if the row was: [10, 5, 4, 10, 1], then the diff would be [-5, -1, 6, -9] and the vector after conversion would be [-1, 0, 1, -1] (notice there is one less column after doing this).
    we call this row_diff
    d. Compute the column difference in intensities. Do this over the direct intensity scores -- not row_diff.
    we call this col_diff
    e. Turn row_diff and col_diff into one long feature vector (row diff and col_diff do not have to be separate RDDs). Row_diff and col_diff are matrices. Flatten them into long vectors, then append them together. Call this features. Row_diff has 50rows x 49columns = 2450 values and col_diff has 49x50 = 2450 values. Thus, features will have 2450 + 2450 = 4900 values,
    f. **Print the feature vectors for: 3677454_2025195.zip-1, 3677454_2025195.zip-18 (just the numpy reduced string output: first 3 + last 3 values)

Example output:

[('3677454_2025195.zip-1', array([ 1, -1, -1, ...,  1, -1, -1])),

 ('3677454_2025195.zip-18', array([-1,  1, -1, ...,  1, -1,  1]))]

Step 3. Use LSH, PCA to find similar images (50 points)

    a. Create a “signature” for each image: Pass the feature vector, in 128 similarly-sized chunks (e.g. a combination of 38 and 39 features per chunk), to an md5 hash function and take one character from the hash per chunk in order to end up with 128 characters as your signature for the image (note that the characters can be bits, ints, letters, or bytes -- your choice). This is analogous to the signature you get per document at the end of minhashing. Here, we are not using minhashing but the next steps for LSH are the same as if we had just produced a signature matrix using minhashing.
    b. Out of all images, run LSH to find approximately 20 candidates that are most similar to images: 3677454_2025195.zip-0, 3677454_2025195.zip-1, 3677454_2025195.zip-18, 3677454_2025195.zip-19. Treat each image as a column.  Note that while you might have things stored as rows being signature vectors for images, the slides depict columns as the signature vectors. 

    	(i) Tweak brands and rows per band in order to get approximately 20 candidates (i.e. anything between 10 to 30 candidates per image is ok).  Note that there are 128 rows total, which you can divide evenly into 1, 2, 4, 8, 16, 32, or 64 bands, but you’re also welcome to divide into a number that doesn’t evenly fit, in which case just leave out the remainder (e.g. 3 bands of 5 rows, and ignore the last row).  
    	(ii) Note that in LSH the columns are signatures. Here each RDD record is a signature, and so from the perspective of LSH, each record is a column.
    	(iii) While pre existing hash code is fine, you must implement LSH otherwise.
    	**print the ~20 candidates for 3677454_2025195.zip-1, 3677454_2025195.zip-18
	    (run on all 4 but only print for these 2)
	    (multiple correct answers)

    c. For each of the candidates from (b), use PCA and find the euclidean difference in low dimensional feature space of the images.

    (i) Start by running PCA across the 4900 dimensional feature vectors of all images to reduce the feature vectors to only 10 dimensions. You must run this step across partitions within spark (use concepts of the course to figure out how to do this; there is no single correct approach and approximate solutions are ok). For example, one option is to run SVD separately on different samples (i.e. random batches) and then combine all the low dimensional batches into one. Solutions that are particularly efficient and accurate may receive extra credit. You may lookup how distributed statistics libraries distribute PCA/SVD but you must abide by policies of academic integrity and not copy code in part or whole.
    (ii) Then, compute the euclidean distance between 3677454_2025195.zip-1, 3677454_2025195.zip-18, and each of their candidates within this 10 dimensional space.  (can be done outside RDDs)
    (iii) **print the distance scores along with their associated imagenames, sorted from least to greatest for each of the candidates you found in 3(b).