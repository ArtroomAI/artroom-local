import React from 'react';
import { Button, Divider, Image, HStack, Text, Menu, MenuButton, MenuList, MenuItem, MenuDivider} from '@chakra-ui/react';
import {
  FaUser,
} from 'react-icons/fa';
import Shards from '../images/shards.png'
const ProfileMenu = ({setLoggedIn}) => {

  return (
    <Menu>
      <MenuButton as={Button} rightIcon={<FaUser/>} olorScheme='teal' variant='outline'>
        <HStack>
          <Image width='10px' src={Shards}/>
          <Text>3000</Text>
          <Divider height='20px' color='white' orientation='vertical' />
          <Text>My Profile</Text>
        </HStack>
      </MenuButton>
      <MenuList>
        <MenuItem>
          Profile
        </MenuItem>
        <MenuItem>
          Settings
        </MenuItem>
        <MenuItem>
          Get More Shards
        </MenuItem>
        <MenuDivider></MenuDivider>
        <MenuItem onClick={()=>setLoggedIn(false)}>
          Logout
        </MenuItem>
      </MenuList>
    </Menu>
  );
}

export default ProfileMenu;