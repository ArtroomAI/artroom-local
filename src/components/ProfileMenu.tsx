import React from 'react'
import * as atom from '../atoms/atoms'
import {
  Button,
  Divider,
  Image,
  HStack,
  Text,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  MenuDivider
} from '@chakra-ui/react'
import { FaUser } from 'react-icons/fa'
import Shards from '../images/shards.png'
import { useRecoilState } from 'recoil'

const ProfileMenu = ({
  setLoggedIn
}: {
  setLoggedIn: React.Dispatch<React.SetStateAction<boolean>>
}) => {
  const [username, setUsername] = useRecoilState(atom.usernameState)
  const [shard, setShard] = useRecoilState(atom.shardState)

  return (
    <Menu>
      <MenuButton
        as={Button}
        leftIcon={<FaUser />}
        colorScheme="teal"
        variant="outline"
      >
        <HStack>
          <Text>{username}</Text>
          <Divider color="white" height="20px" orientation="vertical" />

          <Image src={Shards} width="10px" />

          <div>{shard}</div>
        </HStack>
      </MenuButton>

      <MenuList>
        <MenuItem>Profile</MenuItem>

        <MenuItem>Settings</MenuItem>

        <MenuItem>Get More Shards</MenuItem>

        <MenuDivider />

        <MenuItem onClick={() => setLoggedIn(false)}>Logout</MenuItem>
      </MenuList>
    </Menu>
  )
}

export default ProfileMenu
