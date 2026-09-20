from curl_cffi.requests import AsyncSession
from typing import Optional, Dict, Any, AsyncGenerator, Literal, List
import json
from .pow import DeepSeekPOW
import asyncio
import re
from pathlib import Path
import sys
import subprocess

ThinkingMode = Literal['detailed', 'simple', 'disabled']
SearchMode = Literal['enabled', 'disabled']


class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors"""
    pass


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails"""
    pass


class UploadFilesUnavailable(DeepSeekError):
    """Raised when search enabled"""
    pass


class RateLimitError(DeepSeekError):
    """Raised when API rate limit is exceeded"""
    pass


class NetworkError(DeepSeekError):
    """Raised when network communication fails"""
    pass


class CloudflareError(DeepSeekError):
    """Raised when Cloudflare blocks the request"""
    pass


class APIError(DeepSeekError):
    """Raised when API returns an error response"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"
    MODEL_TYPE = 'default' # 'default' — supports files (only text data, images, and documents)
                           # 'expert'  — does not support files (only prompts)
                           # 'vision'  — supports files (any images and documents)

    def __init__(self, auth_token: str):
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")

        self.auth_token = auth_token
        self.pow_solver = DeepSeekPOW()
        self.last_message_id: Dict[str, Any] = {}
        self._fragment_types: Dict[Any, str] = {}
        self._active_fragment_id: Any = None
        self._current_fragment_type: str = 'RESPONSE'
        self._show_thinking: bool = False
        self._reference_map: Dict[int, Dict[str, Any]] = {}
        self._search_results: List[Dict[str, Any]] = []
        self._marker_to_ref_id: Dict[int, int] = {}
        self._marker_ref_type: Dict[int, str] = {}
        self._tool_fragments: Dict[int, Dict[str, Any]] = {}

        self.session = AsyncSession()

        # Load cookies from JSON file
        cookies_path = Path(__file__).parent / 'dsk' / 'cookies.json'
        
        if not cookies_path.is_file():
            cookies_path.parent.mkdir(parents=True, exist_ok=True)
            open(cookies_path, "w+", encoding='utf8').write("{}")
        
        try:
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\033[93mWarning: Could not load cookies from {cookies_path}: {e}\033[0m", file=sys.stderr)
            self.cookies = {}

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'content-type': 'application/json',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '20241129.1',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '2.5.0',
        }

        if pow_response:
            headers['x-ds-pow-response'] = pow_response

        return headers

    async def _refresh_cookies(self) -> None:
        """Run the cookie refresh script and reload cookies"""
        try:
            # Get path to bypass.py
            script_path = Path(__file__).parent / 'bypass.py'

            # Run the script
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path)
            )
            await proc.communicate()

            # Wait briefly for cookies file to be written
            await asyncio.sleep(2)

            # Reload cookies
            cookies_path = Path(__file__).parent / 'dsk' / 'cookies.json'
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})

        except Exception as e:
            print(f"\033[93mWarning: Failed to refresh cookies: {e}\033[0m", file=sys.stderr)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Dict[str, Any],
        pow_required: bool = False
    ) -> Any:
        url = f"{self.BASE_URL}{endpoint}"

        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                headers = self._get_headers()

                if pow_required:
                    challenge = await self._get_pow_challenge()
                    pow_response = await self.pow_solver.solve_challenge(challenge)
                    headers = self._get_headers(pow_response)

                # Await the request to get the response
                response = await self.session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data,
                    cookies=self.cookies,
                    impersonate='chrome120',
                )
                
                # text is a property, not a method
                text = response.text

                # Cloudflare detection
                if "<!DOCTYPE html>" in text and "Just a moment" in text:
                    print("\033[93mWarning: Cloudflare detected\033[0m", file=sys.stderr)
                    await self._refresh_cookies()
                    retry_count += 1
                    continue

                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(text, response.status_code)
                elif response.status_code != 200:
                    raise APIError(text, response.status_code)

                return json.loads(text)

            except Exception as e:
                if retry_count >= max_retries - 1:
                    raise NetworkError(str(e))
                retry_count += 1

        raise APIError("Failed after retries")

    async def _get_pow_challenge(self) -> Dict[str, Any]:
        try:
            response = await self._make_request(
                'POST',
                '/chat/create_pow_challenge',
                {'target_path': '/api/v0/chat/completion'}
            )
            biz_data = response['data']['biz_data']
            if biz_data is None:
                raise APIError(f"Empty biz_data in challenge response: {response.get('msg', '')}")
            return biz_data['challenge']
        except (KeyError, TypeError):
            raise APIError("Invalid challenge response format from server")

    async def _get_pow_challenge_for_upload(self) -> Dict[str, Any]:
        """Get POW challenge specifically for file upload"""
        try:
            response = await self._make_request(
                'POST',
                '/chat/create_pow_challenge',
                {'target_path': '/api/v0/file/upload_file'}
            )
            biz_data = response['data']['biz_data']
            if biz_data is None:
                raise APIError(f"Empty biz_data in challenge response: {response.get('msg', '')}")
            return biz_data['challenge']
        except (KeyError, TypeError):
            raise APIError("Invalid challenge response format from server")

    async def create_chat_session(self) -> str:
        """Creates a new chat session and returns the session ID"""
        try:
            response = await self._make_request(
                'POST',
                '/chat_session/create',
                {'character_id': None}
            )
            chat_session = response['data']['biz_data']['chat_session']
            if chat_session is None:
                raise APIError(f"Failed to create session: {response.get('msg', '')}")
            return chat_session['id']
        except (KeyError, TypeError):
            raise APIError("Invalid session creation response format from server")

    async def delete_chat_session(self, chat_session_id: str) -> str:
        """Delete current chat session"""
        try:
            await self._make_request(
                'POST',
                '/chat_session/delete',
                {'chat_session_id': chat_session_id}
            )
            return f"Successfully deleted session: {chat_session_id}"
        except KeyError:
            raise APIError("Invalid session delete response format from server")

    async def _upload_single_file(self, file_path: str) -> str:
        """Upload a single file and return its ID"""
        url = f"{self.BASE_URL}/file/upload_file"
        # Get challenge and solve it
        challenge = await self._get_pow_challenge_for_upload()
        pow_response = await self.pow_solver.solve_challenge(challenge)
        
        # Headers for file upload (multipart/form-data)
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '2.5.0',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '2.5.0',
            'x-ds-pow-response': pow_response,
            'x-model-type': self.MODEL_TYPE
        }
        
        retry_count = 0
        max_retries = 2
        
        while retry_count < max_retries:
            try:
                from curl_cffi.requests import AsyncSession
                from curl_cffi import CurlMime

                with open(file_path, "rb") as f:
                    file_data = f.read()

                mp = CurlMime()
                mp.addpart(name="file", data=file_data, filename=Path(file_path).name, content_type="application/octet-stream")

                response = await self.session.post(
                    url,
                    headers=headers,
                    multipart=mp,
                    cookies=self.cookies,
                    impersonate='chrome120',
                )
                # text is a property
                text = response.text
                
                if "<!DOCTYPE html>" in text and "Just a moment" in text:
                    print("\033[93mWarning: Cloudflare detected during upload\033[0m", file=sys.stderr)
                    await self._refresh_cookies()
                    retry_count += 1
                    continue
                
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code != 200:
                    raise APIError(text, response.status_code)
                
                result = json.loads(text)

                if result['data'] is not None:
                    file_id = result['data']['biz_data']['id']
                    # We wait for the file to be ready with a timeout of 60 seconds, polling every 2 seconds.
                    ready_file_id = await self._wait_for_file_ready(file_id, timeout=60.0, poll_interval=2.0)
                    return ready_file_id

            except Exception as e:
                if retry_count >= max_retries - 1:
                    raise NetworkError(f"Failed to upload {file_path}: {str(e)}")
                retry_count += 1
        
        raise APIError(f"Failed to upload {file_path} after retries")

    async def _wait_for_file_ready(self, file_id: str, timeout: float = 60.0, poll_interval: float = 2.0) -> str:
        """
        Wait until uploaded file is successfully processed.
        Returns file_id on success, empty string on failure/timeout.
        """
        start = asyncio.get_event_loop().time()
        last_error = None

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                print(f"\033[93mTimeout waiting for file {file_id} to be ready\033[0m", file=sys.stderr)
                return ''

            try:
                url = f"{self.BASE_URL}/file/fetch_files?file_ids={file_id}"
                challenge = await self._get_pow_challenge()
                pow_response = await self.pow_solver.solve_challenge(challenge)
                headers = self._get_headers(pow_response)

                res = await self.session.get(
                    url,
                    headers=headers,
                    cookies=self.cookies,
                    impersonate='chrome120'
                )
                text = res.text
                if "<!DOCTYPE html>" in text and "Just a moment" in text:
                    print("Cloudflare while polling file status", file=sys.stderr)
                    await self._refresh_cookies()
                    await asyncio.sleep(poll_interval)
                    continue

                result = json.loads(text)

                # Check file ready status
                if result.get('data') and result['data'].get('biz_data'):
                    files = result['data']['biz_data'].get('files', [])
                    if files:
                        status = files[0].get('status')
                        if status == "SUCCESS":
                            return file_id
                        elif status == "PARSING":
                            await asyncio.sleep(poll_interval)
                            continue
                        else:
                            print(f"File {file_id} ended with unexpected status: {status}", file=sys.stderr)
                            return ''
                else:
                    print(f"Unexpected API response while polling file {file_id}", file=sys.stderr)
                    return ''

            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {e}"
            except Exception as e:
                last_error = str(e)

            # Pause before retrying after an error
            print(f"\033[93mError polling file {file_id} (will retry): {last_error}\033[0m", file=sys.stderr)
            await asyncio.sleep(poll_interval)


    async def upload_files(self, file_paths: List[str]) -> List[str]:
        """
        Upload multiple files concurrently and return their IDs
        
        Args:
            file_paths: List of paths to files to upload
            
        Returns:
            List of file IDs in the same order as input
        """
        # Create tasks for concurrent uploads
        tasks = [self._upload_single_file(file_path) for file_path in file_paths]

        # Run all uploads concurrently
        file_ids = await asyncio.gather(*tasks)

        file_ids = [res for res in file_ids if res != '']
        
        return file_ids

    async def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: Optional[str] = None,
        ref_file_ids: Optional[List[str]] = None,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
        show_thinking: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message and get streaming response
        
        Args:
            chat_session_id (str): The ID of the chat session
            prompt (str): The message to send
            parent_message_id (Optional[str]): ID of the parent message for threading
            ref_file_ids (Optional[List[str]]): List of file IDs to reference
            thinking_enabled (bool): Whether to show the thinking process
            search_enabled (bool): Whether to enable web search for up-to-date information
            show_thinking (bool): Whether to include thinking fragments in the output
            
        Returns:
            AsyncGenerator[Dict[str, Any], None]: Yields message chunks with content and type
        """
        
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")
        # if ref_file_ids and search_enabled:
        #     raise UploadFilesUnavailable("To use file uploads, you need to turn off the search.")

        json_data = {
            'chat_session_id': chat_session_id,
            'parent_message_id': self.last_message_id.get(chat_session_id) or parent_message_id,
            'prompt': prompt,
            'ref_file_ids': ref_file_ids if ref_file_ids else [],
            'thinking_enabled': thinking_enabled,
            'search_enabled': search_enabled,
            'model_type': self.MODEL_TYPE,
            'preempt': False,
            'action': None
        }

        # Get challenge and solve it
        challenge = await self._get_pow_challenge()
        pow_response = await self.pow_solver.solve_challenge(challenge)

        headers = self._get_headers(pow_response)

        self._show_thinking = show_thinking
        self._current_fragment_type = 'RESPONSE'
        self._fragment_types = {}
        self._reference_map = {}
        self._search_results = []
        self._active_fragment_id = None
        self._marker_to_ref_id = {}
        self._marker_ref_type = {}
        self._tool_fragments = {}

        # Use async with for stream
        async with self.session.stream(
            'POST',
            f"{self.BASE_URL}/chat/completion",
            headers=headers,
            json=json_data,
            cookies=self.cookies,
            impersonate='chrome120',
        ) as response:

            if response.status_code != 200:
                text = response.text
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                else:
                    print(f"\033[91msearch_query error {response.status_code}: {text[:500]}\033[0m", file=sys.stderr)
                    raise APIError(text, response.status_code)

            async for line in response.aiter_lines():
                # Decode bytes to string if needed
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                
                # Skip empty lines
                if not line or not line.strip():
                    continue
                
                parsed = self._parse_chunk_sync(line)
                if parsed:
                    if parsed.get('type') == 'message_ids':
                        self.last_message_id[chat_session_id] = parsed['response_message_id']
                        continue

                    yield parsed

                    if parsed.get('finish_reason') == 'stop':
                        break

    async def search_query(
        self,
        query: str,
        before_seq_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        json_data = {
            'query': query,
            'before_seq_id': before_seq_id,
        }

        headers = self._get_headers()
        headers['x-client-bundle-id'] = 'com.deepseek.chat'
        headers['x-client-timezone-offset'] = '18000'

        async with self.session.stream(
            'POST',
            f"{self.BASE_URL}/index/query",
            headers=headers,
            json=json_data,
            cookies=self.cookies,
            impersonate='chrome120',
        ) as response:

            if response.status_code != 200:
                text = response.text
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                else:
                    print(f"\033[91msearch_query error {response.status_code}: {text[:500]}\033[0m", file=sys.stderr)
                    raise APIError(text, response.status_code)

            async for line in response.aiter_lines():
                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                line = line.strip()
                if not line:
                    continue

                if line.startswith('event:'):
                    continue

                if not line.startswith('data:'):
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                text_parts = []
                for part in data.get('content', {}).get('parts', []):
                    part_text = part.get('text', '')
                    if part_text:
                        text_parts.append(part_text)

                content = ''.join(text_parts)

                yield {
                    'type': 'text',
                    'content': content,
                    'message_id': data.get('message_id'),
                    'seq_id': data.get('seq_id'),
                    'is_begin': data.get('content', {}).get('is_begin', False),
                    'is_end': data.get('content', {}).get('is_end', False),
                    'is_think': data.get('is_think', False),
                }

    async def continue_stream(
        self,
        chat_session_id: str,
        message_id: Optional[int] = None,
        fallback_to_resume: bool = True,
        show_thinking: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continue a stopped/incomplete stream generation

        Args:
            chat_session_id (str): The ID of the chat session
            message_id (Optional[int]): The response message ID to continue.
                If not provided, uses the last known response_message_id
                tracked from the most recent chat_completion call.
            fallback_to_resume (bool): Whether to fall back to resume if
                the original generation context is no longer available.
            show_thinking (bool): Whether to include thinking fragments in the output

        Returns:
            AsyncGenerator[Dict[str, Any], None]: Yields the same chunk format
            as chat_completion, starting with the accumulated response text
            followed by incremental content.
        """
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

        if message_id is None:
            message_id = self.last_message_id.get(chat_session_id)
        if message_id is None:
            raise ValueError(
                "message_id is required when no stream is tracked for this session"
            )

        self._show_thinking = show_thinking
        self._current_fragment_type = 'RESPONSE'
        self._fragment_types = {}
        self._reference_map = {}
        self._search_results = []
        self._active_fragment_id = None
        self._marker_to_ref_id = {}
        self._marker_ref_type = {}
        self._tool_fragments = {}

        json_data = {
            'chat_session_id': chat_session_id,
            'message_id': message_id,
            'fallback_to_resume': fallback_to_resume,
        }

        headers = self._get_headers()

        async with self.session.stream(
            'POST',
            f"{self.BASE_URL}/chat/continue",
            headers=headers,
            json=json_data,
            cookies=self.cookies,
            impersonate='chrome120',
        ) as response:

            if response.status_code != 200:
                text = response.text
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                else:
                    raise APIError(text, response.status_code)

            async for line in response.aiter_lines():
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                if not line or not line.strip():
                    continue
                if not line.startswith('data:') and not line.startswith('data: '):
                    continue

                data_str = line[6:] if line.startswith('data: ') else line[5:]
                if not data_str or not data_str.strip():
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Handle message IDs
                if 'request_message_id' in data and 'response_message_id' in data:
                    self.last_message_id[chat_session_id] = data['response_message_id']
                    yield {
                        'type': 'message_ids',
                        'response_message_id': data['response_message_id'],
                        'finish_reason': None,
                        'content': '',
                    }
                    continue

                # Skip timestamp updates
                if 'updated_at' in data:
                    continue

                # Handle v.response — full accumulated response state
                v_val = data.get('v')
                if isinstance(v_val, dict) and 'response' in v_val:
                    fragments = v_val['response'].get('fragments', [])

                    self._sync_fragment_tracking(fragments)

                    emitted = self._emit_active_fragment(fragments)
                    if emitted is not None:
                        yield emitted
                    continue

                # Delegate remaining chunks to the standard parser
                parsed = self._parse_chunk_sync(line)
                if parsed:
                    if parsed.get('type') == 'message_ids':
                        self.last_message_id[chat_session_id] = parsed['response_message_id']
                        continue

                    yield parsed

                    if parsed.get('finish_reason') == 'stop':
                        break

    async def stop_stream(
        self,
        chat_session_id: str,
        message_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Stop an active stream generation for a message

        Args:
            chat_session_id (str): The ID of the chat session
            message_id (Optional[int]): The response message ID to stop.
                If not provided, uses the last known response_message_id 
                tracked from the most recent chat_completion call.

        Returns:
            Dict[str, Any]: The API response, e.g.:
            {"code": 0, "msg": "", "data": {"biz_code": 0, "biz_msg": "", "biz_data": None}}
        """
        if message_id is None:
            message_id = self.last_message_id.get(chat_session_id)

        if message_id is None:
            raise ValueError(
                "message_id is required when no active stream is tracked for this session"
            )

        return await self._make_request(
            'POST',
            '/chat/stop_stream',
            {
                'chat_session_id': chat_session_id,
                'message_id': message_id,
            }
        )

    async def index_prepare(self) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/index/prepare"
        headers = self._get_headers()

        response = await self.session.get(
            url,
            headers=headers,
            cookies=self.cookies,
            impersonate='chrome120',
        )

        if response.status_code != 200:
            text = response.text
            if response.status_code == 401:
                raise AuthenticationError("Invalid or expired authentication token")
            elif response.status_code == 429:
                raise RateLimitError("API rate limit exceeded")
            else:
                raise APIError(text, response.status_code)

        return json.loads(response.text)

    async def list_conversations(
        self,
        pinned: bool = False,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch list of chat sessions/conversations.

        Args:
            pinned: Filter by pinned status (default False).
            cursor: Optional pagination cursor value (e.g. a seq_id) for
                    fetching the next page.

        Returns:
            List of chat session objects.
        """
        params = f"lte_cursor.pinned={str(pinned).lower()}"
        if cursor is not None:
            params += f"&lte_cursor.value={cursor}"
        url = f"{self.BASE_URL}/chat_session/fetch_page?{params}"

        headers = self._get_headers()
        headers['x-client-bundle-id'] = 'com.deepseek.chat'
        headers['x-client-locale'] = 'en_US'
        headers['x-client-platform'] = 'web'
        headers['x-client-timezone-offset'] = '18000'
        headers['x-client-version'] = '2.2.0'

        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                response = await self.session.get(
                    url,
                    headers=headers,
                    cookies=self.cookies,
                    impersonate='chrome120',
                )

                text = response.text

                if "<!DOCTYPE html>" in text and "Just a moment" in text:
                    print("\033[93mWarning: Cloudflare detected\033[0m", file=sys.stderr)
                    await self._refresh_cookies()
                    retry_count += 1
                    continue

                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(text, response.status_code)
                elif response.status_code != 200:
                    raise APIError(text, response.status_code)

                result = json.loads(text)
                return result.get('data', {}).get('biz_data', {}).get('chat_sessions', [])

            except Exception as e:
                if retry_count >= max_retries - 1:
                    raise NetworkError(str(e))
                retry_count += 1

        raise APIError("Failed to fetch conversations after retries")

    async def get_history(self, convo_id: str) -> Dict[str, Any]:
        """Fetch full conversation history"""
        url = f"{self.BASE_URL}/chat/history_messages?chat_session_id={convo_id}"

        # Get challenge and solve it
        challenge = await self._get_pow_challenge()
        pow_response = await self.pow_solver.solve_challenge(challenge)
        
        headers = self._get_headers(pow_response)

        response = await self.session.get(
            url,
            headers=headers,
            cookies=self.cookies
        )

        if response.status_code != 200:
            return {
                "error": response.status_code,
                "detail": response.text
            }

        return json.loads(response.text)

    def _extract_batch_content(self, ops: Any) -> Optional[str]:
        """Extract the textual content from a BATCH operations list"""
        if not isinstance(ops, list):
            return None
        parts: List[str] = []
        for op in ops:
            if (isinstance(op, dict)
                    and op.get('p') == 'content'
                    and isinstance(op.get('v'), str)
                    and op['v'].strip()):
                parts.append(op['v'])
        return ''.join(parts) if parts else None

    def _sync_fragment_tracking(self, fragments: Any) -> None:
        """Update fragment type map and the active fragment from a fragments list.

        The active fragment is the one currently receiving content appends.
        It is inferred as the fragment with the most recent non-empty content
        (falling back to the last created fragment), because the fragment array
        in accumulated state snapshots may already contain future fragments.
        """
        if not isinstance(fragments, list) or not fragments:
            return

        for fr in fragments:
            if isinstance(fr, dict) and fr.get('id') is not None:
                self._fragment_types[fr['id']] = fr.get('type', 'RESPONSE')
                refs = fr.get('references')
                if isinstance(refs, list):
                    self._update_references(refs)
                self._register_tool_fragment(fr)

        last_non_empty = None
        for fr in fragments:
            if isinstance(fr, dict) and fr.get('content'):
                last_non_empty = fr.get('id')

        last_created = None
        if isinstance(fragments[-1], dict):
            last_created = fragments[-1].get('id')

        active_id = last_non_empty if last_non_empty is not None else last_created
        if active_id is not None:
            self._active_fragment_id = active_id
            self._current_fragment_type = self._fragment_types.get(active_id, 'RESPONSE')

    def _update_references(self, refs: Any) -> None:
        """Accumulate reference metadata keyed by reference id"""
        if not isinstance(refs, list):
            return
        for ref in refs:
            if isinstance(ref, dict) and ref.get('id') is not None:
                rid = ref['id']
                if rid not in self._reference_map:
                    self._reference_map[rid] = {}
                self._reference_map[rid].update(ref)

    def _register_tool_fragment(self, fr: Dict[str, Any]) -> None:
        """Remember TOOL_OPEN/TOOL_SEARCH fragments by id.

        A citation reference {id, type} points at one of these fragments.
        TOOL_OPEN fragments carry the actual source URL in their `result`
        field, which is what we link to.
        """
        ftype = fr.get('type')
        if ftype not in ('TOOL_OPEN', 'TOOL_SEARCH'):
            return
        if fr.get('id') is None:
            return
        self._tool_fragments[fr['id']] = fr

    def _map_markers_to_refs(self, content: str, refs: List[Dict[str, Any]]) -> None:
        """Map [reference:N] markers to reference objects carried by the same BATCH op.

        The marker number in the content is just a sequential citation counter;
        the reference {id, type} points to the TOOL_OPEN/TOOL_SEARCH fragment
        (identified by `id`) that the citation refers to.
        """
        if not content or not refs:
            return
        markers = [int(m) for m in re.findall(r'\[reference:(\d+)\]', content)]
        for marker, ref in zip(markers, refs):
            rid = ref.get('id')
            if rid is None:
                continue
            self._marker_to_ref_id[marker] = rid
            self._marker_ref_type[marker] = ref.get('type', '')

    def _clean_references(self, content: str) -> str:
        """Replace [reference:N] markers with markdown links [[N]](url), or drop them"""
        if not content or '[reference:' not in content:
            return content

        def repl(m):
            marker = int(m.group(1))
            rid = self._marker_to_ref_id.get(marker)
            rtype = self._marker_ref_type.get(marker, '')
            link = None
            if rid is not None:
                # 1) explicit url from reference metadata (keyed by reference id)
                ref = self._reference_map.get(rid)
                if ref and ref.get('url'):
                    link = ref['url']
                # 2) TOOL_OPEN fragment carries the actual source URL
                elif rtype == 'TOOL_OPEN' and rid in self._tool_fragments:
                    result = self._tool_fragments[rid].get('result') or {}
                    link = result.get('url')
            if not link:
                return ''
            rendered = f'[[{marker}]]({link})'
            # keep a separating space before the citation link
            if m.start() > 0 and content[m.start() - 1] not in ' \t\n\r':
                rendered = ' ' + rendered
            return rendered

        return re.sub(r'\[reference:(\d+)\]', repl, content)

    def _update_search_results(self, results: Any, op: Optional[str] = None) -> None:
        """Accumulate search results. Always extend; reference ids are global
        1-based indices into the combined list across all query result sets."""
        if not isinstance(results, list):
            return
        self._search_results.extend(results)

    def _sync_content_fragment(self, fragment_id: Any) -> None:
        """Mark the fragment receiving incremental content appends as active"""
        if fragment_id is None:
            return
        self._active_fragment_id = fragment_id
        if fragment_id in self._fragment_types:
            self._current_fragment_type = self._fragment_types[fragment_id]

    def _current_type(self) -> str:
        """Type of the currently active fragment"""
        if self._active_fragment_id is not None:
            return self._fragment_types.get(self._active_fragment_id, self._current_fragment_type)
        return self._current_fragment_type

    def _emit_active_fragment(self, fragments: Any) -> Optional[Dict[str, Any]]:
        """Return a content chunk for the active fragment, or None if it should be skipped"""
        active_id = self._active_fragment_id
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            if active_id is not None and fragment.get('id') != active_id:
                continue

            content = fragment.get('content')
            if not isinstance(content, str) or not content.strip():
                return None

            content = self._clean_references(content)
            if not content:
                return None

            ftype = fragment.get('type', 'RESPONSE')
            if ftype == 'RESPONSE':
                return {
                    'type': 'text',
                    'content': content,
                    'finish_reason': None
                }
            if ftype == 'THINK':
                if not self._show_thinking:
                    return None
                return {
                    'type': 'thinking',
                    'content': content,
                    'finish_reason': None
                }
            return None
        return None

    def _parse_chunk_sync(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Parse a SSE chunk from the API response (synchronous version)"""
        if not chunk:
            return None

        try:
            # Handle data: lines
            if chunk.startswith('data: '):
                data_str = chunk[6:]
            elif chunk.startswith('data:'):
                data_str = chunk[5:]
            else:
                # Skip non-data lines (like event: lines)
                return None

            # Skip empty data
            if not data_str or not data_str.strip():
                return None

            # Parse JSON
            data = json.loads(data_str)

            # Handle nested response format (accumulated snapshot with full structure)
            if 'v' in data and isinstance(data['v'], dict) and 'response' in data['v']:
                response_data = data['v']['response']
                fragments = response_data.get('fragments', [])

                # Collect any top-level references
                self._update_references(response_data.get('references'))

                # Update fragment tracking (active fragment + type map)
                self._sync_fragment_tracking(fragments)

                emitted = self._emit_active_fragment(fragments)
                if emitted is not None:
                    return emitted

                # Check for message IDs in the response structure
                if 'message_id' in response_data:
                    return {
                        'type': 'message_ids',
                        'response_message_id': response_data['message_id'],
                        'finish_reason': None,
                        'content': ''
                    }

            # Handle new fragment append (response/fragments) to track its type
            if data.get('p') == 'response/fragments' and data.get('o') == 'APPEND':
                fragments = data.get('v', [])
                if isinstance(fragments, list) and fragments:
                    self._sync_fragment_tracking(fragments)

                    fragment = fragments[-1]
                    new_id = fragment.get('id')
                    # The newly appended fragment becomes the active one being written
                    if new_id is not None:
                        self._active_fragment_id = new_id
                        self._current_fragment_type = fragment.get('type', 'RESPONSE')

                    content = fragment.get('content') or ''
                    if isinstance(content, str):
                        content = self._clean_references(content)
                    if content.strip():
                        ftype = fragment.get('type')
                        if ftype == 'THINK':
                            if not self._show_thinking:
                                return None
                            return {
                                'type': 'thinking',
                                'content': content,
                                'finish_reason': None
                            }
                        return {
                            'type': 'text',
                            'content': content,
                            'finish_reason': None
                        }
                return None

            # Handle chunks with just 'v' field (simplified format)
            if 'v' in data and 'p' not in data and isinstance(data['v'], str):
                ftype = self._current_type()
                if ftype == 'THINK' and not self._show_thinking:
                    return None
                return {
                    'type': 'thinking' if ftype == 'THINK' else 'text',
                    'content': self._clean_references(str(data['v'])),
                    'finish_reason': None
                }

            # Handle BATCH operations carrying a list of relative sub-operations
            # (e.g. content + references metadata when a search reference is inserted)
            if data.get('o') == 'BATCH' and isinstance(data.get('v'), list):
                ref_objs: List[Dict[str, Any]] = []
                for op in data['v']:
                    if not isinstance(op, dict):
                        continue
                    if op.get('p') == 'references' and isinstance(op.get('v'), list):
                        self._update_references(op['v'])
                        for r in op['v']:
                            if isinstance(r, dict) and r.get('id') is not None:
                                ref_objs.append(r)
                    elif op.get('p') == 'results' and isinstance(op.get('v'), list):
                        self._update_search_results(op['v'], op.get('o'))
                    elif op.get('p') == 'fragments' and isinstance(op.get('v'), list):
                        for fr in op['v']:
                            if isinstance(fr, dict):
                                self._register_tool_fragment(fr)

                content = self._extract_batch_content(data['v']) or ''
                self._map_markers_to_refs(content, ref_objs)
                content = self._clean_references(content)
                if not content:
                    return None

                ftype = self._current_type()
                if ftype == 'THINK' and not self._show_thinking:
                    return None

                return {
                    'type': 'thinking' if ftype == 'THINK' else 'text',
                    'content': content,
                    'finish_reason': None
                }

            # Handle reference metadata appends
            fragment_refs_match = re.match(r'^response/fragments/(-?\d+)/references$', data.get('p', ''))
            if fragment_refs_match and isinstance(data.get('v'), list):
                self._update_references(data['v'])
                return None

            # Handle search results (carry the actual URLs for references)
            fragment_results_match = re.match(r'^response/fragments/(-?\d+)/results$', data.get('p', ''))
            if fragment_results_match and isinstance(data.get('v'), list):
                self._update_search_results(data['v'], data.get('o'))
                return None

            content_match = re.match(r'^response/fragments/(-?\d+)/content$', data.get('p', ''))
            if content_match and 'v' in data:
                fragment_id = content_match.group(1)
                if fragment_id != '-1':
                    self._sync_content_fragment(int(fragment_id))

                v_value = data.get('v')
                if not isinstance(v_value, str):
                    return None

                ftype = self._current_type()
                if ftype == 'THINK' and not self._show_thinking:
                    return None

                return {
                    'type': 'thinking' if ftype == 'THINK' else 'text',
                    'content': self._clean_references(v_value),
                    'finish_reason': None
                }

            # Handle finished status
            if data.get('p') == 'response/status' and data.get('v') == 'FINISHED':
                return {
                    'type': 'text',
                    'content': '',
                    'finish_reason': 'stop'
                }

            # Handle message IDs (first message)
            if 'request_message_id' in data and 'response_message_id' in data:
                return {
                    'type': 'message_ids',
                    'response_message_id': data['response_message_id'],
                    'finish_reason': None,
                    'content': ''
                }

            # Skip other message types
            return None

        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"Warning: Error parsing chunk: {e}", file=sys.stderr)
            return None
            
    async def close(self):
        """Close the async session"""
        await self.session.close()
