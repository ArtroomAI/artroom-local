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
function QueueRow (props) {
    const { index, id, prompt, negative_prompt, ddim_steps, n_iter, H, W, scale, seed, ckpt, sampler, init_image, strength, mask, invert, lastItem } = props;

    const textColor = useColorModeValue(
        'gray.700',
        'white'
    );
    return (
        <Tr>
            <Td
                border={lastItem
                    ? 'none'
                    : null}
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
                        minWidth="100%"
                        noOfLines={1}>
                        {index}
                    </Text>
                </Flex>
            </Td>

            <Td
                border={lastItem
                    ? 'none'
                    : null}
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
                        minWidth="100%"
                        noOfLines={1}>
                        {prompt}
                    </Text>
                </Flex>
            </Td>

            <Td
                border={lastItem
                    ? 'none'
                    : null}
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
                        minWidth="100%"
                        noOfLines={1}>
                        {ckpt}
                    </Text>
                </Flex>
            </Td>

            {W === 0
                ? <Td
                    border={lastItem
                        ? 'none'
                        : null}
                    borderBottomColor="#56577A">
                    <Text
                        color="#fff"
                        fontSize="sm"
                        fontWeight="bold"
                        pb=".5rem">
                        Init Image
                    </Text>
                </Td>
                : <Td
                    border={lastItem
                        ? 'none'
                        : null}
                    borderBottomColor="#56577A">
                    <Text
                        color="#fff"
                        fontSize="sm"
                        fontWeight="bold"
                        pb=".5rem">
                        {W}

                        x

                        {H}
                    </Text>
                </Td>}

            <Td
                border={lastItem
                    ? 'none'
                    : null}
                borderBottomColor="#56577A">
                <Text
                    color="#fff"
                    fontSize="sm"
                    fontWeight="bold"
                    pb=".5rem">
                    x
                    {n_iter}
                </Text>
            </Td>

            <Td
                border={lastItem
                    ? 'none'
                    : null}
                borderBottomColor="#56577A">
                <HStack>
                    <QueueModal props={props} />

                    <RemoveFromQueue index={index} />
                </HStack>
            </Td>
        </Tr>
    );
}

export default QueueRow;
