local beam = require 's2sa.beam'

function main()
  beam.init(arg)
  local opt = beam.getOptions()

  while true do
    local line = io.read()
    local modifications = io.read()

    -- Read out modification format
    result, nbests = beam.search(line, json.decode(modifications))

    print(result)
  end
end

main()
