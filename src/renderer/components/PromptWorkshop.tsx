import React, { useState } from 'react'
import {
  Box,
  Text,
  Select,
  Button,
} from '@chakra-ui/react'

import {
  batchNameState
} from '../SettingsManager'
import { useRecoilState } from 'recoil';

function PromptWorkshop() {
  const [selectedOption, setSelectedOption] = useState('');
  const [batchName, setBatchName] = useRecoilState(batchNameState)
  const handleOptionChange = (event: { target: { value: React.SetStateAction<string>; }; }) => {
    setSelectedOption(event.target.value);
  };
  const updateBatchName = () => {
    // Update the batchName to "Banana"
    setBatchName('Banana');
  };
    return (
    <Box height="90%" ml="30px" p={4} rounded="md" width="90%">
      <Text>{batchName}</Text>
      <Text>Select an option:</Text>
      {/* Dropdown menu */}
      <Select value={selectedOption} onChange={handleOptionChange}>
        <option value="option1">Option 1</option>
        <option value="option2">Option 2</option>
        <option value="option3">Option 3</option>
      </Select>
            {/* Button to update batchName */}
            <Button onClick={updateBatchName}>Update batchName to Banana</Button>

    </Box>
  )
}

export default PromptWorkshop
