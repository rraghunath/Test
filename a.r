#load required library
# First clear the workspace
rm(list = ls())
library(ggplot2)
library(class)

#################################################
# PREPROCESSING
#################################################

data <- iris                # create copy of iris dataframe
labels <- data$Species      # store labels
data$Species <- NULL        # remove labels from feature set (note: could
                            # alternatively use neg indices on column index in knn call)

#################################################
# TRAIN/TEST SPLIT
#################################################

set.seed(1)         # initialize random seed for consistency
                    # NOTE -- run for various seeds --> need for CV!

test.pct <- 0.1    # pct of data to use for test set
N <- nrow(data)     # total number of records (150)
for (mm in 1:10)
{
  test.index <- sample(1:N, replace = FALSE, test.pct * N)       # random sample of records (test set)
  test.data <- data[test.index, ] # get the test data first
  train.data <- data[-test.index, ]       # perform train/test split

  test.labels <- as.factor(as.matrix(labels)[test.index, ])     # extract test set labels
  train.labels <- as.factor(as.matrix(labels)[-test.index, ])     # extract training set labels
#################################################
# APPLY MODEL
#################################################

  err.rates <- data.frame()       # initialize results object

  max.k <- 15
  for (k in 1:max.k)              # perform fit for various values of k
  {
      knn.fit <- knn(train = train.data,          # training set
                      test = test.data,           # test set
                      cl = train.labels,          # true labels
                      k = k                       # number of NN to poll
                 )

      cat('\n', 'k = ', k, ', test.pct = ', test.pct, '\n', sep='')     # print params
      print(table(test.labels, knn.fit))          # print confusion matrix

      this.err <- sum(test.labels != knn.fit) / length(test.labels)    # store gzn err
      err.rates <- rbind(err.rates, this.err)     # append err to total results
  }

#################################################
# OUTPUT RESULTS
#################################################

  results <- data.frame(1:max.k, err.rates)   # create results summary data frame
  names(results) <- c('k', 'err.rate')        # label columns of results df

# create title for results plot
  title <- paste('knn results (test.pct = ', test.pct, 'CVFold = ', mm, ')', sep='')

# create results plot
  results.plot <- ggplot(results, aes(x=k, y=err.rate)) + geom_point() + geom_line()
  results.plot <- results.plot + ggtitle(title)

# draw results plot (note need for print stmt inside script to draw ggplot)
  print(results.plot)
}
#################################################
# NOTES
#################################################

# what happens for high values (eg 100) of max.k? have a look at this plot:
# > results.plot <- ggplot(results, aes(x=k, y=err.rate)) + geom_smooth()

# our implementation here is pretty naive, meant to illustrate concepts rather
# than to be maximally efficient...see alt impl in DMwR package (with feature
# scaling):
#
# > install.packages('DMwR')
# > library(DMwR)
# > knn

# ed. note: how not to do it (black box)
# http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Classification/kNN

# R docs
# http://cran.r-project.org/web/packages/class/class.pdf
# http://cran.r-project.org/web/packages/DMwR/DMwR.pdf
