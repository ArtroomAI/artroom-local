import {
  Flex,
  IconButton,
  Td,
  Text,
  Tr,
  useColorModeValue,
  HStack
} from '@chakra-ui/react';
import React from 'react';
import RemoveFromQueue from './RemoveFromQueue';
import QueueModal from './QueueModal';
function QueueRow(props) {
  const {index, id, prompt, negative_prompt, ddim_steps, n_iter, H, W, scale, seed, ckpt, sampler, init_image, strength, mask, invert, lastItem} = props;

  const textColor = useColorModeValue('gray.700', 'white');
  return (
    <Tr>
      <Td
        borderBottomColor='#56577A'
        border={lastItem ? 'none' : null}>
        <Flex align='center' py='.8rem' minWidth='100%' flexWrap='nowrap'>
            <Text noOfLines={1} fontSize='sm' color='#fff' fontWeight='normal' minWidth='100%'>
              {index}
            </Text>        
        </Flex>
      </Td>
      <Td
        minWidth={{ sm: '250px' }}
        ps='0px'
        borderBottomColor='#56577A'
        border={lastItem ? 'none' : null}>
        <Flex align='center' py='.8rem' minWidth='100%' flexWrap='nowrap'>
            <Text noOfLines={1} fontSize='sm' color='#fff' fontWeight='normal' minWidth='100%'>
              {prompt}
            </Text>        
        </Flex>
      </Td>
      <Td
        minWidth={{ sm: '250px' }}
        ps='0px'
        borderBottomColor='#56577A'
        border={lastItem ? 'none' : null}>
        <Flex align='center' py='.8rem' minWidth='100%' flexWrap='nowrap'>
            <Text noOfLines={1} fontSize='sm' color='#fff' fontWeight='normal' minWidth='100%'>
              {ckpt}
            </Text>        
        </Flex>
      </Td>
      {W === 0 ? 
      <Td borderBottomColor='#56577A' border={lastItem ? 'none' : null}>
        <Text fontSize='sm' color='#fff' fontWeight='bold' pb='.5rem'>
          Init Image
        </Text>
      </Td>
        :
      <Td borderBottomColor='#56577A' border={lastItem ? 'none' : null}>
        <Text fontSize='sm' color='#fff' fontWeight='bold' pb='.5rem'>
          {W}x{H}
        </Text>
      </Td>
      }

      <Td borderBottomColor='#56577A' border={lastItem ? 'none' : null}>
      <Text fontSize='sm' color='#fff' fontWeight='bold' pb='.5rem'>
          x{n_iter}
        </Text>
      </Td>
      <Td borderBottomColor='#56577A' border={lastItem ? 'none' : null}>
        <HStack>
          <QueueModal props={props}/>
          <RemoveFromQueue index={index}/>
        </HStack>  
      </Td>
    </Tr>
  );
}

export default QueueRow;
