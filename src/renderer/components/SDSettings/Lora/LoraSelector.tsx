import React, { useEffect, useRef, useState } from 'react'
import {
  VStack,
  HStack,
  Input,
  IconButton,
  Icon,
  Text,
  Flex,
  FormControl,
  FormLabel,
} from '@chakra-ui/react'
import { FaTimes } from 'react-icons/fa'
import { IoMdCloud } from 'react-icons/io'
import { useRecoilState } from 'recoil'
import { loraState } from '../../../SettingsManager'
import ReactDOM from 'react-dom'

function LoraSelector({ options, cloudMode }: { options: any[]; cloudMode: boolean }) {
  const [lora, setLora] = useRecoilState(loraState)
  const [searchQuery, setSearchQuery] = useState('')
  const [inputFocus, setInputFocus] = useState(false)
  const filteredOptions = searchQuery.length
    ? options.filter((item) => item.toLowerCase().includes(searchQuery.toLowerCase()))
    : options
  const containerRef = useRef(null) // Ref for the container div
  const [menuCoordinates, setMenuCoordinates] = useState({ top: 0, left: 0 })

  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      setMenuCoordinates({ top: rect.bottom, left: rect.left })
    }
  }, [containerRef])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      setSelectedIndex((prevIndex) => Math.min(prevIndex + 1, filteredOptions.length - 1))
    } else if (e.key === 'ArrowUp') {
      setSelectedIndex((prevIndex) => Math.max(prevIndex - 1, 0))
    } else if (e.key === 'Enter' && selectedIndex > -1) {
      handleAddItem({ name: filteredOptions[selectedIndex], weight: 1 })
    }
  }

  const handleSearch = (event) => {
    const searchQuery = event.target.value
    setSearchQuery(searchQuery)
  }

  const handleAddItem = (item: { name: string; weight: number }) => {
    if (!lora.some((i) => i.name === item.name)) {
      setLora([...lora, { ...item, weight: 1 }])
      setSearchQuery('')
    }
  }

  const handleRemoveItem = (itemToRemove: { name: string; weight: number }) => {
    setLora(lora.filter((item) => item.name !== itemToRemove.name))
  }

  const handleweightChange = (itemToUpdate: { name: string; weight: number }, weight: number) => {
    const newItems = lora.map((item) => {
      if (item.name === itemToUpdate.name) {
        return { ...item, weight }
      }
      return item
    })
    setLora(newItems)
  }

  return (
    <FormControl className="lora-input">
      <FormLabel htmlFor="Lora">
        <HStack>
          <Text>Choose Your Lora</Text>
          {cloudMode ? <Icon as={IoMdCloud} /> : null}
        </HStack>
      </FormLabel>
      <div ref={containerRef} style={{ position: 'relative' }}>
        <Input
          type="text"
          placeholder="Search by filename..."
          value={searchQuery}
          onChange={handleSearch}
          onFocus={() => setInputFocus(true)}
          onBlur={() => setInputFocus(false)}
        />
        {inputFocus &&
          ReactDOM.createPortal(
            <VStack
              position="absolute"
              top={`${menuCoordinates.top}px`}
              left={`${menuCoordinates.left}px`}
              zIndex="1000"
              bg="#080B16"
              boxShadow="md"
              borderRadius="md"
              w="275px"
              maxH="300px"
              overflowY="auto"
            >
              {filteredOptions.map((item) => (
                <Flex
                  maxWidth="100%"
                  key={item}
                  p="8px"
                  cursor="pointer"
                  onMouseDown={() => handleAddItem({ name: item, weight: 1 })}
                >
                  <Text textAlign="left" maxWidth="100%">
                    {item}
                  </Text>
                </Flex>
              ))}
            </VStack>,
            document.body // Render directly in the body element
          )}
      </div>

      <VStack pt={4}>
        {lora.length > 0 && (
          <HStack>
            <Text fontWeight="normal" fontSize="sm" width="120px">
              Name
            </Text>
            <Text fontWeight="normal" fontSize="sm" width="70px">
              Weight
            </Text>
          </HStack>
        )}
        {lora.map((item) => (
          <Flex align="center" key={item.name}>
            <Text fontWeight="normal" width="120px" pr="15px">
              {item.name}
            </Text>
            <Input
              width="70px"
              type="number"
              min={0}
              value={item.weight}
              onChange={(event) => handleweightChange(item, parseFloat(event.target.value))}
            />
            <IconButton
              aria-label="Remove"
              icon={<Icon as={FaTimes} />}
              variant="ghost"
              size="sm"
              onClick={() => handleRemoveItem(item)}
            />
          </Flex>
        ))}
      </VStack>
    </FormControl>
  )
}

export default LoraSelector
