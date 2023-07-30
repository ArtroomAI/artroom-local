import React, { useState } from 'react';
import { Text, Box, Input, Button, Flex, FormControl, FormLabel, VStack } from '@chakra-ui/react';
import { useRecoilState } from 'recoil';
import { authenticatedState } from '../../SettingsManager';

const Authentication = ({ Component, correctPassword, authenticatedKey }: { Component: any, correctPassword: string, authenticatedKey :string }) => {
  const [password, setPassword] = useState('');
  const [authenticated, setAuthenticated] = useRecoilState(authenticatedState)

  const handlePasswordChange = (event: { target: { value: React.SetStateAction<string>; }; }) => {
    setPassword(event.target.value);
  };

  const checkPassword = () => {
    // Replace 'YOUR_CORRECT_PASSWORD' with the actual password you want to use
    if (password === correctPassword) {
      setAuthenticated({...authenticated, [authenticatedKey] : true});
    } else {
      alert('Incorrect password! Please try again.');
    }
  };

  return (
    <>
      {!authenticated ? (  
        <Box width='100%' minHeight='80vh' display='flex' justifyContent='center' alignItems='center'>
          <VStack display='flex' justifyContent='center' alignItems='center' maxW={'500px'}>
            <Text fontSize='xl' mx='auto' textAlign='center' justifyContent='center'>{`Discover our latest feature in early access! Join us on Patreon as an early subscriber to try it out :)`}</Text>
            <Box p={4} mx="auto">
              <FormControl>
                <FormLabel>Early Access Code:</FormLabel>
                <Flex align="center">
                  <Input
                    type="string"
                    value={password}
                    onChange={handlePasswordChange}
                    mr={2}
                  />
                  <Button onClick={checkPassword} colorScheme="blue">
                    Submit
                  </Button>
                </Flex>
              </FormControl>
              <Box mt={2}>
                  <Button colorScheme="teal" variant="link" onClick={()=>window.api.openPatreon()}>
                     {`Don't Have One Yet? Click Here!`}
                  </Button>
              </Box>
            </Box>
          </VStack>
        </Box>
      ) : (
        <Component />
      )}
    </>
  );
};

export default Authentication;
