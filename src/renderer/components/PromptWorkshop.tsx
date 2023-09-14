import React, { useState } from 'react';
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
  Spacer,
  Tooltip,
  Image
} from '@chakra-ui/react';

import {
  DEFAULT_NEGATIVE_PROMPT,
  batchNameState
} from '../SettingsManager';
import { useRecoilState } from 'recoil';
import { FaQuestionCircle } from 'react-icons/fa';
import DragDropFile from './DragDropFile/DragDropFile';
import { AutoResizeTextarea } from './Prompt';

function PromptWorkshop() {
  const [selectedOption1, setSelectedOption1] = useState('');
  const [selectedOption2, setSelectedOption2] = useState('');
  const [batchName, setBatchName] = useRecoilState(batchNameState);
  const [inputValue1, setInputValue1] = useState('');
  const [inputValue2, setInputValue2] = useState('');

  const handleOptionChange1 = (event) => {
    setSelectedOption1(event.target.value);
    // Clear the input value when changing the selected option
    setInputValue1('');
  };

  const handleOptionChange2 = (event) => {
    setSelectedOption2(event.target.value);
    // Clear the input value when changing the selected option
    setInputValue2('');
  };


  const renderInput1 = () => {
    switch (selectedOption1) {
      case 'Prompt':
      case 'Negative Prompt':
        return (
          <FormControl>
            <FormLabel>{selectedOption1}:</FormLabel>
            <Input
              value={inputValue1}
              onChange={(e) => setInputValue1(e.target.value)}
            />
          </FormControl>
        );
      case 'Seed':
      case 'Width':
      case 'Height':
      case 'CFG Scale':
      case 'Steps':
      case 'Image Strength':
      case 'Clip Skip':
        return (
          <FormControl>
            <FormLabel>{selectedOption1}:</FormLabel>
            <Input
              type="number"
              value={inputValue1}
              onChange={(e) => setInputValue1(e.target.value)}
            />
          </FormControl>
        );
      case 'Lora':
      case 'Model':
      case 'VAE':
      case 'Controlnet':
      case 'Remove Background':
      case 'Sampler':
        return (
          <FormControl>
            <FormLabel>{selectedOption1}:</FormLabel>
            {/* You can replace this with a dropdown component */}
            <Select>
              <option value="option1">Option 1</option>
              <option value="option2">Option 2</option>
              <option value="option3">Option 3</option>
            </Select>
          </FormControl>
        );
      default:
        return null; // Return null for unsupported options
    }
  };

  const renderInput2 = () => {
    // Similar logic as renderInput1 for selectedOption2
    switch (selectedOption2) {
      case 'Prompt':
      case 'Negative Prompt':
        return (
          <FormControl>
            <FormLabel>{selectedOption2}:</FormLabel>
            <Input
              value={inputValue2}
              onChange={(e) => setInputValue2(e.target.value)}
            />
          </FormControl>
        );
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
              type="number"
              value={inputValue2}
              onChange={(e) => setInputValue2(e.target.value)}
            />
          </FormControl>
        );
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
        );
      default:
        return null; // Return null for unsupported options
    }
  };


  return (
    <VStack width="100%">
      <HStack alignItems="flex-start" justifyContent="flex-start" width="100%">
        <Flex align="center" justify="center">
          {/* Dropdown menu 1 */}
          <Box mr={4}>
            <Text>Select option 1:</Text>
            <Select value={selectedOption1} onChange={handleOptionChange1}>
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

          {renderInput1()}
        </Flex>

        {/* Image */}
        <Image
          src="https://i.imgur.com/VEaD2Pr.png"
          alt="Your Image"
          boxSize="200px"
        />
      </HStack>

      {/* Selected Option 2 */}
      <VStack alignItems="flex-start">
        {/* Dropdown menu 2 */}
          <Box mt={4}>
            <Text>Select option 2:</Text>
            <Select value={selectedOption2} onChange={handleOptionChange2}>
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

        {renderInput2()}
      </VStack>

      <Button height="90%" ml="20px" p={4} rounded="md" width="50%">
        <Text>Generate XY-Plot</Text>
      </Button>
    </VStack>
  );
}

export default PromptWorkshop;
