import {
    Flex,
    Td,
    Text,
    Tr
} from '@chakra-ui/react';
import React from 'react';
function QueueModalRow (props) {
    return (
        <Tr>
            <Td
                border="none"
                borderBottomColor="#56577A">
                <Flex
                    align="center"
                    flexWrap="nowrap"
                    minWidth="100%"
                    py=".8rem">
                    <Text
                        color="#fff"
                        fontSize="sm"
                        fontWeight="normal"
                        minWidth="100%">
                        {props.name}
                    </Text>
                </Flex>
            </Td>

            <Td
                border="none"
                borderBottomColor="#56577A"
                minWidth={{ sm: '250px' }}
                ps="0px">
                <Flex
                    align="center"
                    flexWrap="nowrap"
                    minWidth="100%"
                    py=".8rem">
                    <Text
                        color="#fff"
                        fontSize="sm"
                        fontWeight="normal"
                        minWidth="100%">
                        {props.value}
                    </Text>
                </Flex>
            </Td>
        </Tr>
    );
}

export default QueueModalRow;

