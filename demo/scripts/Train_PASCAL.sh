#!/usr/bin/env bash
# utilities
function init { 
    if [ -z "${!1}" ]; then export $1=$2; fi
}
function define { 
    export $1=$2
}
function waiting {
    for pid in "$@"
    do
        while [ -e /proc/$pid ]
        do
            sleep 1
        done
    done
}

# global settings
init enableSave true
init notify "false"
init maxEpoch 20
init epochStep '10'
init learningRateDecayRatio 0.1
init removeOldCheckpoints false
init optimMethod "sgd"
init learningRate 0.01
init batchSize 64
init gpuDevice "{1,2,3,4}"

# tasks
function SP-VGGNetVOC2007 {
    define dataset "VOC"
    define year "2007"
    define trainset "trainval"
    define valset "test"
    define testset "test"
    define model "SP-VGGNet"
    define savePath "logs/SP-VGGNet-VOC2007"
    define note "VGGNet-with-SoftProposal"
    th train.lua
}

# run tasks
SP-VGGNetVOC2007
