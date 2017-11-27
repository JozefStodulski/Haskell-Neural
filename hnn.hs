module Main where

type Inputs = [Float]
type Weight = Float
type Node = [Weight]
type Layer = [Node]
type Network = [Layer]
type Sample = ([Float],[Float])
type Samples = [Sample]
bias = 1 :: Float

main :: IO ()
main = do
    putStrLn "Create (1), Train (2) or Run (3) the network?"
    option <- getLine
    case option of
        "1" -> do
            putStrLn "Network dimentions:"
            dimentions' <- getLine
            let dimentions = reverse . map read $ words dimentions' :: [Int]
            print . reverse $ createNetwork dimentions
        "2" -> do
            putStrLn "Network [Layer [Node [Float]]]:"
            network' <- getLine
            let network = read network' :: Network
            putStrLn "Training data:"
            samples' <- getLine
            let samples = read samples' :: Samples
            putStrLn "Learning rate:"
            learningRate' <- getLine
            let learningRate = read learningRate' :: Float
            putStrLn "Passes:"
            passes' <- getLine
            let passes = read passes' :: Int
            putStrLn $ train network samples learningRate passes
        "3" -> do
            putStrLn "Network [Layer [Node [Float]]]:"
            network' <- getLine
            let network = read network' :: Network
            putStrLn "Inputs [Float]:"
            inputs' <- getLine
            let inputs = read inputs' :: Inputs
            putStrLn $ run network inputs
    main


-- FORWARD
run :: Network -> Inputs -> String
run network inputs =
    if nInputs == nGivenInputs
        then show . reverse $ propagate (reverse network) inputs
        else "Network expects " ++ show nInputs " inputs. You provided " ++ show nGivenInputs
    where
        (nInputs, nGivenInputs) = (length $ head network, length inputs + 1)

propagate :: Network -> Inputs -> [Float]
propagate (layer:previousLayers) inputs
    | previousLayers == [] = map (nodeValue [] inputs) layer
    | otherwise            = map (nodeValue previousLayers inputs) layer

nodeValue :: Network -> Inputs -> Node -> Float   
nodeValue previousLayers inputs nodeWeights
    | previousLayers == [] = transfer $ zipWith (*) (bias : inputs) nodeWeights
    | otherwise            = transfer $ zipWith (*) (bias : propagate previousLayers inputs) nodeWeights

transfer :: (Floating a) => [a] -> a
transfer = sigmoid . sum

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (-1 * x))

sigmoid' :: (Floating a) => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)


-- BACKWARD
train :: Network -> Samples -> Float -> Int -> String   -- to do: calculate batches
train network _ _ 0                       = show network
train network samples learningRate epochs = train network' samples learningRate (epochs-1)
    where
        network' = backpropagate network networkErrors learningRate
        networkErrors = map (0.5 / (fromIntegral (length samples)) *) sumTotalErrors network samples

sumTotalErrors :: Network -> Samples -> [Float]
sumTotalErrors network ((inputs,targets):samples)
    | samples == [] = sumSquareDifferences
    | otherwise     = zipWith (+) sumSquareDifferences (sumTotalErrors network samples)
    where
        sumSquareDifferences = zipWith (\o t->(o-t)**2) outputs targets
        outputs = propagate network inputs

backpropagate :: Network -> [Float] -> Float -> Network       -- accepts reverse network
backpropagate network@(layer:previousLayers) networkErrors learningRate = [[[4]]]    -- temporary
--    | previousLayers == [] = layer' : []
--    | otherwise            = layer' : backpropagate previousLayers previousLayerErrors learningRate
--    where
--        layer' = 
--        previousLayerErrors =
--        gradient =
-- NOTE: GRADIENT OF NODE IS NODE GRADIENT * LOWER GRADIENT


--backpropagate :: Network -> Samples -> Float -> Network  --only works on output layer. need to recalculate internal errors
--backpropagate network@(layer:previousLayers) samples learningRate
--    | previousLayers == [] = layer' : []
--    | otherwise            = layer' : backpropagate previousLayers samples learningRate
--    where
--        layerErrors = map (1 / (2 *  fromIntegral (length samples)) *) (errors network samples)
--        layer' = map (propagateNode learningRate layerErrors) layer

--propagateNode :: Float -> [Float] -> Node -> Node
--propagateNode learningRate errr weights =
--    zipWith (\e w -> learningRate * e * w) errr weights


-- CREATE
createNetwork :: [Int] -> Network       -- accepts reverse network
createNetwork (nNodes:rest@(nWeights:dimentions))
    | dimentions == [] = replicate nNodes (replicate nWeights 4) : []
    | otherwise        = replicate nNodes (replicate nWeights 4) : createNetwork rest


-- network dimentions
-- 1 2 2

-- XOR network -- 2 input nodes, 2 hidden, 1 output. first weight of each node is bias.
-- [[[-10,20,20],[30,-20,-20]],[[-30,20,20]]]
-- untrained
-- [[[4,5,6],[7,8,9]],[[1,2,3]]]

-- inputs
-- [0,0] [0,1] [1,0] [1,1]

-- sample data
-- [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
