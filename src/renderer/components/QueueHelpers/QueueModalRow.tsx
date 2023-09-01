import React from 'react'
import { Flex, Td, Text, Tr, Image } from '@chakra-ui/react'

function QueueModalRow(props: { name: string; value: string | number | typeof Image }) {
  return (
    <Tr>
      <Td border="none">
        <Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
          <Text color="#fff" fontSize="sm" fontWeight="normal" minWidth="100%">
            {props.name}
          </Text>
        </Flex>
      </Td>

      <Td border="none" minWidth={{ sm: '250px' }} ps="0px">
        <Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
          {typeof props.value !== 'object' ? (
            <Text color="#fff" fontSize="sm" fontWeight="normal" minWidth="100%">
              {props.value as number | string}
            </Text>
          ) : (
            props.value
          )}
        </Flex>
      </Td>
    </Tr>
  )
}

export default QueueModalRow
