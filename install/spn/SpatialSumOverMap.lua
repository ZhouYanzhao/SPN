local SpatialSumOverMap, parent = torch.class('nn.SpatialSumOverMap', 'nn.Module')

function SpatialSumOverMap:__init()
    parent.__init(self)
    self.gpucompatible = true
end

function SpatialSumOverMap:updateOutput(input)
    self.output = input:sum(4):sum(3)
    return self.output
end

function SpatialSumOverMap:updateGradInput(input, gradOutput)
    assert(input:nDimension() == 4, 'only supports batch mode')
    self.gradInput = gradOutput:reshape(input:size(1), input:size(2), 1, 1)
    self.gradInput = self.gradInput:expandAs(input)
    return self.gradInput
end
