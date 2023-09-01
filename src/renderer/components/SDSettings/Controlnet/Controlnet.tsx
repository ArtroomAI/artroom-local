import React, { useContext } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import * as atom from '../../../atoms/atoms'
import { VStack, Button, Checkbox, FormControl, FormLabel, HStack, Select } from '@chakra-ui/react'
import {
  controlnetState,
  initImageState,
  modelsDirState,
  usePreprocessedControlnetState,
} from '../../../SettingsManager'
import { SocketContext } from '../../../socket'
import ControlnetPreview from './ControlnetPreview'

const Controlnet = () => {
  const socket = useContext(SocketContext)
  const [controlnet, setControlnet] = useRecoilState(controlnetState)
  const [controlnetPreview, setControlnetPreview] = useRecoilState(atom.controlnetPreviewState)
  const models_dir = useRecoilValue(modelsDirState)
  const [usePreprocessedControlnet, setUsePreprocessedControlnet] = useRecoilState(
    usePreprocessedControlnetState
  )
  const initImage = useRecoilValue(initImageState)

  return (
    <VStack>
      <FormControl className="controlnet-input">
        <HStack>
          <FormLabel htmlFor="Controlnet">Choose your controlnet</FormLabel>
        </HStack>
        <HStack>
          <Select
            id="controlnet"
            name="controlnet"
            onChange={(event) => setControlnet(event.target.value)}
            value={controlnet}
            variant="outline"
          >
            <option value="none">None</option>

            <option value="depth">Depth</option>

            <option value="normal">Normal</option>

            <option value="pose">Pose</option>

            <option value="hed">HED</option>

            <option value="softedge">Soft Edge</option>

            <option value="canny">Canny</option>

            <option value="mlsd">MLSD</option>

            <option value="lineart">Lineart</option>

            <option value="lineart_anime">Lineart Anime</option>

            <option value="scribble">Scribble</option>

            <option value="ip2p">Instruct Pix2Pix</option>

            <option value="inpaint">Inpaint</option>

            <option value="shuffle">Shuffle</option>

            <option disabled={true} value="tile">
              Tile
            </option>
          </Select>
          <Button
            variant="outline"
            disabled={controlnet === 'none' || usePreprocessedControlnet}
            onClick={() => {
              socket.emit('preview_controlnet', { initImage, controlnet, models_dir })
            }}
          >
            Preview
          </Button>
        </HStack>
      </FormControl>
      <HStack>
        <FormLabel htmlFor="use_random_seed">Use Preprocessed ControlNet</FormLabel>
        <Checkbox
          id="use_preprocessed_controlnet"
          isChecked={usePreprocessedControlnet}
          name="use_preprocessed_controlnet"
          onChange={() => {
            setUsePreprocessedControlnet((usePreprocessedControlnet) => !usePreprocessedControlnet)
          }}
          pb="12px"
        />
      </HStack>

      {controlnetPreview.length > 0 && <ControlnetPreview />}
    </VStack>
  )
}

export default Controlnet
