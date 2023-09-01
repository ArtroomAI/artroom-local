import React from 'react'
import {
  VStack,
  HStack,
  FormControl,
  FormLabel,
  Checkbox,
  Tooltip,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  NumberInput,
  NumberInputField,
} from '@chakra-ui/react'
import { useRecoilState } from 'recoil'
import {
  highresStepsState,
  highresStrengthState,
  highresfixOnlyState,
} from '../../../SettingsManager'

export default function HighresUpscale() {
  const [highresfixOnly, setHighresfixOnly] = useRecoilState(highresfixOnlyState)
  const [highresSteps, setHighresSteps] = useRecoilState(highresStepsState)
  const [highresStrength, setHighresStrength] = useRecoilState(highresStrengthState)

  return (
    <VStack>
      <FormControl className="highres_steps">
        <FormLabel htmlFor="highres_steps">Highres steps:</FormLabel>
        <NumberInput
          id="highres_steps"
          name="highres_steps"
          onChange={setHighresSteps}
          value={highresSteps}
          variant="outline"
        >
          <NumberInputField id="highres_steps" />
        </NumberInput>
      </FormControl>

      <FormControl className="highres_strength">
        <FormLabel htmlFor="highres_strength">Highres strength:</FormLabel>
        <Slider
          defaultValue={0.15}
          id="highres_strength"
          max={1}
          min={0.0}
          name="highres_strength"
          onChange={setHighresStrength}
          step={0.01}
          value={highresStrength}
          variant="outline"
        >
          <SliderTrack bg="#EEEEEE">
            <SliderFilledTrack bg="#4f8ff8" />
          </SliderTrack>

          <Tooltip
            bg="#4f8ff8"
            color="white"
            isOpen={true}
            label={`${highresStrength}`}
            placement="right"
          >
            <SliderThumb />
          </Tooltip>
        </Slider>
      </FormControl>

      <Tooltip label="Upscale using diffusion upscale, also known as highres fix. The process is more GPU intense than regular upscale and much slower!">
        <HStack>
          <FormLabel htmlFor="highresfix_only">Only upscale</FormLabel>

          <Checkbox
            id="highresfix_only"
            isChecked={highresfixOnly}
            name="highresfix_only"
            onChange={() => {
              setHighresfixOnly((e) => !e)
            }}
            pb="12px"
          />
        </HStack>
      </Tooltip>
    </VStack>
  )
}
