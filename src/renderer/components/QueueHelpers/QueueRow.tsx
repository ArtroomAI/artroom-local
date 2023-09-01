import React from 'react'
import { Flex, Td, Text, Tr, HStack, IconButton } from '@chakra-ui/react'
import RemoveFromQueue from './RemoveFromQueue'
import QueueModal from './QueueModal'

import { useSortable } from '@dnd-kit/sortable'

import { CSS } from '@dnd-kit/utilities'
import { BsArrowsExpand } from 'react-icons/bs'

export function SortableItem({
  id,
  index,
  arr,
  row,
}: {
  id: string
  index: number
  arr: QueueType[]
  row: QueueType
}) {
  const { listeners, setNodeRef, setActivatorNodeRef, transform, transition } = useSortable({
    id: id,
  })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  }

  if (index === 0) {
    return (
      <Tr key={index}>
        <Td
          border={index === arr.length - 1 ? 'none' : null}
          borderBottomColor="#56577A"
          maxWidth="100px"
          px={0}
        ></Td>
        <QueueRow index={index + 1} lastItem={index === arr.length - 1} {...row} />
      </Tr>
    )
  }

  return (
    <Tr ref={setNodeRef} style={style} key={index}>
      <Td
        border={index === arr.length - 1 ? 'none' : null}
        borderBottomColor="#56577A"
        maxWidth="100px"
        px={0}
      >
        <IconButton
          aria-label="drag"
          icon={<BsArrowsExpand />}
          ref={setActivatorNodeRef}
          {...listeners}
        />
      </Td>
      <QueueRow index={index + 1} lastItem={index === arr.length - 1} {...row} />
    </Tr>
  )
}

function QueueRow(props: QueueTypeWithIndex) {
  return (
    <>
      <Td border={props.lastItem ? 'none' : null} borderBottomColor="#56577A">
        <Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
          <Text color="#fff" fontSize="sm" fontWeight="normal" minWidth="100%" noOfLines={1}>
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
          <Text color="#fff" fontSize="sm" fontWeight="normal" minWidth="100%" noOfLines={1}>
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
          <Text color="#fff" fontSize="sm" fontWeight="normal" minWidth="100%" noOfLines={1}>
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
    </>
  )
}

export default QueueRow
