import React, { useState } from 'react'
import {
  Box,
  Text,
  Select,
  Button,
  HStack,
  VStack,
  Flex,
  FormControl,
  FormLabel,
  Input,
  Image,
  CheckboxGroup,
  Checkbox,
} from '@chakra-ui/react'
import XYPlot from '../images/XYPlot.png'

function PromptWorkshop() {
  const [selectedOption1, setSelectedOption1] = useState('Prompt')
  const [selectedOption2, setSelectedOption2] = useState('Prompt')
  const [inputValue1, setInputValue1] = useState('')
  const [inputValue2, setInputValue2] = useState('')
  const [parsedArray1, setParsedArray1] = useState([])
  const [parsedArray2, setParsedArray2] = useState([])
  const [isInvalidInput1, setIsInvalidInput1] = useState(false)
  const [isInvalidInput2, setIsInvalidInput2] = useState(false)

  const handleOptionChange =
    (setter: (arg0: any) => void) => (event: { target: { value: any } }) => {
      setter(event.target.value)
    }

  const parseInput = (setterArray, setterInvalid) => (input) => {
    try {
      if (!input.length) return
      const arr = `${input},`
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item !== '')
      setterArray(arr)
      setterInvalid(false)
    } catch (err) {
      setterInvalid(true)
    }
  }
  const parseInput1 = parseInput(setParsedArray1, setIsInvalidInput1)
  const parseInput2 = parseInput(setParsedArray2, setIsInvalidInput2)

  const renderInput = (
    selectedOption,
    inputValue,
    setterInput,
    parseInputFunction,
    parsedArray,
    isInvalidInput
  ) => {
    switch (selectedOption) {
      case 'Prompt':
      case 'Negative Prompt':
        return (
          <FormControl>
            <FormLabel>Enter your {selectedOption}, separate with ",":</FormLabel>
            <Input
              value={inputValue}
              onChange={(e) => {
                parseInputFunction(e.target.value)
                setterInput(e.target.value)
              }}
              isInvalid={isInvalidInput}
            />
            {isInvalidInput && <Text color="red">Invalid input</Text>}
            <Flex mt={3}>
              {parsedArray.length > 0 &&
                parsedArray.map((value, index) => (
                  <Button key={index} onClick={() => {}}>
                    {value}
                  </Button>
                ))}
            </Flex>
          </FormControl>
        )

      case 'Seed':
      case 'Width':
      case 'Height':
      case 'CFG Scale':
      case 'Steps':
      case 'Image Strength':
      case 'Clip Skip':
        return (
          <FormControl>
            <FormLabel>Enter your {selectedOption}, separate with ",":</FormLabel>
            <Input
              value={inputValue}
              onChange={(e) => {
                parseInputFunction(e.target.value)
                setterInput(e.target.value)
              }}
              isInvalid={isInvalidInput}
            />
            {isInvalidInput && <Text color="red">Invalid input</Text>}
            <Flex mt={3}>
              {parsedArray.map((value, index) => (
                <Button key={index} onClick={() => {}}>
                  {value} X
                </Button>
              ))}
            </Flex>
          </FormControl>
        )
      case 'Lora':
      case 'Model':
      case 'VAE':
      case 'Controlnet':
      case 'Remove Background':
      case 'Sampler':
        return (
          <FormControl>
            <FormLabel>Please select from {selectedOption}:</FormLabel>
            {/* Use CheckboxGroup for multiple selections */}
            <CheckboxGroup value={selectedOption} onChange={parseInputFunction}>
              <VStack alignItems="flex-start">
                <Checkbox value="option1">Option 1</Checkbox>
                <Checkbox value="option2">Option 2</Checkbox>
                <Checkbox value="option3">Option 3</Checkbox>
              </VStack>
            </CheckboxGroup>
          </FormControl>
        )
      default:
        return null // Return null for unsupported options
    }
  }

  return (
    <VStack width="100%">
      <HStack alignItems="flex-start" justifyContent="flex-start" width="100%">
        <VStack align="start" justify="start" width="30%">
          <Box mr={4}>
            <Text>Y-Axis:</Text>
            <Select value={selectedOption1} onChange={handleOptionChange(setSelectedOption1)}>
              <option value="Prompt">Prompt</option>
              <option value="Negative Prompt">Negative Prompt</option>
              <option value="Seed">Seed</option>
              <option value="Lora">Lora</option>
              <option value="Model">Model</option>
              <option value="VAE">VAE</option>
              <option value="Controlnet">Controlnet</option>
              <option value="Remove Background">Remove Background</option>
              <option value="Width">Width</option>
              <option value="Height">Height</option>
              <option value="CFG Scale">CFG Scale</option>
              <option value="Steps">Steps</option>
              <option value="Sampler">Sampler</option>
              <option value="Image Strength">Image Strength</option>
              <option value="Clip Skip">Clip Skip</option>
            </Select>
          </Box>

          {renderInput(
            selectedOption1,
            inputValue1,
            setInputValue1,
            parseInput1,
            parsedArray1,
            isInvalidInput1
          )}
        </VStack>

        {/* Image */}
        <Image src={XYPlot} alt="Your Image" boxSize="400px" />
      </HStack>

      {/* Selected Option 2 */}
      <VStack alignItems="flex-start" width="45%">
        <Box mt={4}>
          <Text>X-axis:</Text>
          <Select value={selectedOption2} onChange={handleOptionChange(setSelectedOption2)}>
            <option value="Prompt">Prompt</option>
            <option value="Negative Prompt">Negative Prompt</option>
            <option value="Seed">Seed</option>
            <option value="Lora">Lora</option>
            <option value="Model">Model</option>
            <option value="VAE">VAE</option>
            <option value="Controlnet">Controlnet</option>
            <option value="Remove Background">Remove Background</option>
            <option value="Width">Width</option>
            <option value="Height">Height</option>
            <option value="CFG Scale">CFG Scale</option>
            <option value="Steps">Steps</option>
            <option value="Sampler">Sampler</option>
            <option value="Image Strength">Image Strength</option>
            <option value="Clip Skip">Clip Skip</option>
          </Select>
        </Box>
        {renderInput(
          selectedOption2,
          inputValue2,
          setInputValue2,
          parseInput2,
          parsedArray2,
          isInvalidInput2
        )}{' '}
      </VStack>

      <Button height="90%" ml="20px" p={4} rounded="md" width="50%">
        <Text>Generate XY-Plot</Text>
      </Button>
    </VStack>
  )
}

export default PromptWorkshop
