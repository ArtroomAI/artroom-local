import {React, useState} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import Prompt from './Prompt';
import UnifiedCanvasDisplay from './InvokeAI/UnifiedCanvas/UnifiedCanvasDisplay';
import {
    Box,
    Flex,
    Button,
    VStack,
    createStandaloneToast,
  } from '@chakra-ui/react';

function InvokeAI() {
    const { ToastContainer, toast } = createStandaloneToast()
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [focused, setFocused] = useState(false);

    return(
        <Flex transition='all .25s ease' ml={navSize === 'large' ? '180px' : '100px'} align='center' justify='center' width='100%'> 
            <Box width='100%' align='center'>
                <VStack spacing={4} align='center'>
                    <UnifiedCanvasDisplay />
                    <Button className='run-button' ml={2} width='250px'>'Run'</Button>
                    <Box width='80%'>
                    <Prompt setFocused={setFocused}/>
                    </Box>
                </VStack>
            </Box>
        </Flex>
    )
}
export default InvokeAI;
