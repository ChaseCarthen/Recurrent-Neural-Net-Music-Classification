require 'DatasetLoader'

dl = DatasetLoader("processed","au","audio")
dl:loadTraining()
dl:loadNextSet()
