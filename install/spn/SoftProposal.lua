local SoftProposal, parent = torch.class('nn.SoftProposal', 'nn.Module')

function SoftProposal:__init(factor, tolerance, maxIteration)
    parent.__init(self)
    self.output = torch.Tensor()
    self.proposal = torch.Tensor()
    self.proposalBuffer = torch.Tensor()
    self.transferMatrix = torch.Tensor()
    self.distanceMetric = torch.Tensor()
    self.mW = 0
    self.mH = 0
    self.N = 0
    self.factor = factor
    self.tolerance = tolerance or 0.0001
    self.maxIteration = maxIteration or 20
end

local function makeContiguous(self, input, gradOutput)
    if not input:isContiguous() then
        self._input = self._input or input.new()
        self._input:resizeAs(input):copy(input)
        input = self._input
    end
    if gradOutput then
       if not gradOutput:isContiguous() then
            self._gradOutput = self._gradOutput or gradOutput.new()
            self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
            gradOutput = self._gradOutput
       end
    end
    return input, gradOutput
end

function SoftProposal:lazyInit(input)
    self.nBatch = input:size(1)
    self.nChannel = input:size(2)
    self.mH = input:size(3)
    self.mW = input:size(4)
    self.N = self.mH * self.mW
    self.transferMatrix:resize(self.N, self.N)
    self.proposal:resize(self.nBatch, self.mH, self.mW)
    self.proposalBuffer:resize(self.mH, self.mW)
    self.factor = self.factor or self.N * 0.15
    -- init distance metric
    self.distanceMetric:resize(self.N, self.N)
    input.spn.SP_InitDistanceMetric(self, self.distanceMetric, self.factor, self.mW, self.mH, self.N)
end

function SoftProposal:updateOutput(input)
    assert(input:nDimension() == 4, 'only supports batch mode')
    assert(input:type() == 'torch.CudaTensor', 'only supports GPU mode')
    if self.nBatch ~= input:size(1) or 
        self.nChannel ~= input:size(2) or 
        self.mH ~= input:size(3) or 
        self.mW ~= input:size(4) then
        self:lazyInit(input)
    end
    input = makeContiguous(self, input)
    self.output:resizeAs(input)
    -- generate soft proposal
    input.spn.SP_Generate(
        self,
        input,
        self.distanceMetric,
        self.transferMatrix,
        self.proposal,
        self.proposalBuffer,
        self.nBatch,
        self.nChannel,
        self.mW,
        self.mH,
        self.N,
        self.tolerance,
        self.maxIteration)
    -- couple with CNN activation
    input.spn.SP_Couple(
        self,
        input,
        self.proposal,
        self.output,
        self.nBatch,
        self.nChannel,
        self.N)
    return self.output
end

function SoftProposal:updateGradInput(input, gradOutput)
    if self.gradInput then
        input, gradOutput = makeContiguous(self, input, gradOutput)
        self.gradInput:resizeAs(input)
        -- -- couple with error signal
        input.spn.SP_Couple(
            self,
            gradOutput,
            self.proposal,
            self.gradInput,
            self.nBatch,
            self.nChannel,
            self.N)
        return self.gradInput
    end
end

function SoftProposal:__tostring__()
    return string.format('%s(%d, %d)', torch.type(self), self.mW, self.mH)
end
