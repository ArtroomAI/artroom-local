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

  const handleOptionChange1 = (event) => {
    setSelectedOption1(event.target.value)
    // Clear the input value when changing the selected option
    setInputValue1('')
  }

  const handleOptionChange2 = (event) => {
    setSelectedOption2(event.target.value)
    // Clear the input value when changing the selected option
    setInputValue2('')
  }

  const renderInput1 = () => {
    switch (selectedOption1) {
      case 'Prompt':
      case 'Negative Prompt':
        return (
          <FormControl>
            <FormLabel>Enter your {selectedOption1}, separate with ",":</FormLabel>
            <Input value={inputValue1} onChange={(e) => setInputValue1(e.target.value)} />
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
            <FormLabel>Enter your {selectedOption1}, separate with ",":</FormLabel>
            <Input
              type="text"
              value={inputValue1}
              onChange={(e) => setInputValue1(e.target.value)}
            />
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
            <FormLabel>Please select from {selectedOption1}:</FormLabel>
            {/* Use CheckboxGroup for multiple selections */}
            <CheckboxGroup value={selectedOption1} onChange={setSelectedOption1}>
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

  const renderInput2 = () => {
    // Similar logic as renderInput1 for selectedOption2
    switch (selectedOption2) {
      case 'Prompt':
      case 'Negative Prompt':
        return (
          <FormControl>
            <FormLabel>{selectedOption2}:</FormLabel>
            <Input value={inputValue2} onChange={(e) => setInputValue2(e.target.value)} />
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
            <FormLabel>{selectedOption2}:</FormLabel>
            <Input
              type="text"
              value={inputValue2}
              onChange={(e) => setInputValue2(e.target.value)}
            />
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
            <FormLabel>{selectedOption2}:</FormLabel>
            {/* You can replace this with a dropdown component */}
            <Select>
              <option value="option1">Option 1</option>
              <option value="option2">Option 2</option>
              <option value="option3">Option 3</option>
            </Select>
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
            <Select value={selectedOption1} onChange={handleOptionChange1}>
              <option value="Prompt">Prompt +++</option>
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

          {renderInput1()}
        </VStack>

        {/* Image */}
        <Image src={XYPlot} alt="Your Image" boxSize="400px" />
      </HStack>

      {/* Selected Option 2 */}
      <VStack alignItems="flex-start" width="45%">
        <Box mt={4}>
          <Text>X-axis:</Text>
          <Select value={selectedOption2} onChange={handleOptionChange2}>
            <option value="Prompt">Prompt ***</option>
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

        {renderInput2()}
      </VStack>

      <Button height="90%" ml="20px" p={4} rounded="md" width="50%">
        <Text>Generate XY-Plot</Text>
      </Button>
    </VStack>
  )
}

export default PromptWorkshop
