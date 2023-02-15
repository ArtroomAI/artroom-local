import { Flex, Td, Text, Tr, HStack } from '@chakra-ui/react'
import React from 'react'
import RemoveFromQueue from './RemoveFromQueue'
import QueueModal from './QueueModal'

function QueueRow(props: QueueTypeWithIndex) {
  return (
    <Tr>
      <Td border={props.lastItem ? 'none' : null} borderBottomColor="#56577A">
        <Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
          <Text
            color="#fff"
            fontSize="sm"
            fontWeight="normal"
            minWidth="100%"
            noOfLines={1}
          >
            {props.index}
          </Text>
        </Flex>
      </Td>

      <Td
        border={props.lastItem ? 'none' : null}
        borderBottomColor="#56577A"
        minWidth={{ sm: '250px' }}
        ps="0px"
      >
        <Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
          <Text
            color="#fff"
            fontSize="sm"
            fontWeight="normal"
            minWidth="100%"
            noOfLines={1}
          >
            {props.text_prompts}
          </Text>
        </Flex>
      </Td>

      <Td
        border={props.lastItem ? 'none' : null}
        borderBottomColor="#56577A"
        minWidth={{ sm: '250px' }}
        ps="0px"
      >
        <Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
          <Text
            color="#fff"
            fontSize="sm"
            fontWeight="normal"
            minWidth="100%"
            noOfLines={1}
          >
            {props.ckpt}
          </Text>
        </Flex>
      </Td>

      {props.width === 0 ? (
        <Td border={props.lastItem ? 'none' : null} borderBottomColor="#56577A">
          <Text color="#fff" fontSize="sm" fontWeight="bold" pb=".5rem">
            Init Image
          </Text>
        </Td>
      ) : (
        <Td border={props.lastItem ? 'none' : null} borderBottomColor="#56577A">
          <Text color="#fff" fontSize="sm" fontWeight="bold" pb=".5rem">
            {props.width}x{props.height}
          </Text>
        </Td>
      )}

      <Td border={props.lastItem ? 'none' : null} borderBottomColor="#56577A">
        <Text color="#fff" fontSize="sm" fontWeight="bold" pb=".5rem">
          x{props.n_iter}
        </Text>
      </Td>

      <Td border={props.lastItem ? 'none' : null} borderBottomColor="#56577A">
        <HStack>
          <QueueModal {...props} />

          <RemoveFromQueue index={props.index} />
        </HStack>
      </Td>
    </Tr>
  )
}

export default QueueRow
