require 'Encoder'
AudioEncoder = torch.class("AudioEncoder","Encoder")

function AudioEncoder:__init()
	Encoder.__init()
end

function AudioEnocoder:forward(data)
end