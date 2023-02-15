import React from 'react'
import { useRecoilState } from 'recoil'
import * as atom from '../atoms/atoms'
import { FaDiscord } from 'react-icons/fa'
import {
  MenuButton,
  HStack,
  Icon,
  Text,
  Flex,
  Menu,
  Link,
  Tooltip
} from '@chakra-ui/react'
const Discord = () => {
  const [navSize, changeNavSize] = useRecoilState(atom.navSizeState)
  return (
    <>
      <Flex
        alignItems={navSize === 'small' ? 'center' : 'flex-start'}
        flexDir="column"
        mt={15}
        onClick={window.api.openDiscord}
        w="100%"
      >
        <Menu placement="right">
          <Link
            _hover={{ textDecor: 'none', backgroundColor: '#AEC8CA' }}
            borderRadius={8}
            p={2.5}
          >
            <Tooltip
              fontSize="md"
              label={navSize === 'small' ? 'Join Discord' : ''}
              placement="bottom"
              shouldWrapChildren
            >
              <MenuButton
                bg="transparent"
                className="discord-link"
                width="100%"
              >
                <HStack>
                  <Icon
                    as={FaDiscord}
                    color="#82AAAD"
                    fontSize="xl"
                    justifyContent="center"
                  />

                  <Text
                    align="center"
                    display={navSize === 'small' ? 'none' : 'flex'}
                    fontSize="m"
                    pl={5}
                    pr={10}
                  >
                    Join Discord
                  </Text>
                </HStack>
              </MenuButton>
            </Tooltip>
          </Link>
        </Menu>
      </Flex>
    </>
  )
}

export default Discord
