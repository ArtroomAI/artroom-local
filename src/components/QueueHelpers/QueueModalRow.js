import {
    Flex,
    Td,
    Text,
    Tr,
  } from '@chakra-ui/react';
  import React from 'react';
  function QueueModalRow(props) {  
    return (
        <Tr>
        <Td
            borderBottomColor='#56577A'
            border='none'>
            <Flex align='center' py='.8rem' minWidth='100%' flexWrap='nowrap'>
                <Text fontSize='sm' color='#fff' fontWeight='normal' minWidth='100%'>
                {props.name}
                </Text>        
            </Flex>
        </Td>
        <Td
            minWidth={{ sm: '250px' }}
            ps='0px'
            borderBottomColor='#56577A'
            border='none'>
            <Flex align='center' py='.8rem' minWidth='100%' flexWrap='nowrap'>
                <Text fontSize='sm' color='#fff' fontWeight='normal' minWidth='100%'>
                {props.value}
                </Text>        
            </Flex>
        </Td>
    </Tr>
    );
  }
  
  export default QueueModalRow;
  