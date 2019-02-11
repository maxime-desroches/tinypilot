/*  =========================================================================
    zhashx - extended generic type-free hash container

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZHASHX_H_INCLUDED__
#define __ZHASHX_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif


//  @warning THE FOLLOWING @INTERFACE BLOCK IS AUTO-GENERATED BY ZPROJECT
//  @warning Please edit the model at "api/zhashx.api" to make changes.
//  @interface
//  This is a stable class, and may not change except for emergencies. It
//  is provided in stable builds.
//  This class has draft methods, which may change over time. They are not
//  in stable releases, by default. Use --enable-drafts to enable.
//  This class has legacy methods, which will be removed over time. You
//  should not use them, and migrate any code that is still using them.
// Destroy an item
typedef void (zhashx_destructor_fn) (
    void **item);

// Duplicate an item
typedef void * (zhashx_duplicator_fn) (
    const void *item);

// Compare two items, for sorting
typedef int (zhashx_comparator_fn) (
    const void *item1, const void *item2);

// compare two items, for sorting
typedef void (zhashx_free_fn) (
    void *data);

// compare two items, for sorting
typedef size_t (zhashx_hash_fn) (
    const void *key);

// Serializes an item to a longstr.                       
// The caller takes ownership of the newly created object.
typedef char * (zhashx_serializer_fn) (
    const void *item);

// Deserializes a longstr into an item.                   
// The caller takes ownership of the newly created object.
typedef void * (zhashx_deserializer_fn) (
    const char *item_str);

// Callback function for zhashx_foreach method.                              
// This callback is deprecated and you should use zhashx_first/_next instead.
typedef int (zhashx_foreach_fn) (
    const char *key, void *item, void *argument);

//  Create a new, empty hash container
CZMQ_EXPORT zhashx_t *
    zhashx_new (void);

//  Unpack binary frame into a new hash table. Packed data must follow format
//  defined by zhashx_pack. Hash table is set to autofree. An empty frame    
//  unpacks to an empty hash table.                                          
CZMQ_EXPORT zhashx_t *
    zhashx_unpack (zframe_t *frame);

//  Destroy a hash container and all items in it
CZMQ_EXPORT void
    zhashx_destroy (zhashx_t **self_p);

//  Insert item into hash table with specified key and item.               
//  If key is already present returns -1 and leaves existing item unchanged
//  Returns 0 on success.                                                  
CZMQ_EXPORT int
    zhashx_insert (zhashx_t *self, const void *key, void *item);

//  Update or insert item into hash table with specified key and item. If the
//  key is already present, destroys old item and inserts new one. If you set
//  a container item destructor, this is called on the old value. If the key 
//  was not already present, inserts a new item. Sets the hash cursor to the 
//  new item.                                                                
CZMQ_EXPORT void
    zhashx_update (zhashx_t *self, const void *key, void *item);

//  Remove an item specified by key from the hash table. If there was no such
//  item, this function does nothing.                                        
CZMQ_EXPORT void
    zhashx_delete (zhashx_t *self, const void *key);

//  Delete all items from the hash table. If the key destructor is  
//  set, calls it on every key. If the item destructor is set, calls
//  it on every item.                                               
CZMQ_EXPORT void
    zhashx_purge (zhashx_t *self);

//  Return the item at the specified key, or null
CZMQ_EXPORT void *
    zhashx_lookup (zhashx_t *self, const void *key);

//  Reindexes an item from an old key to a new key. If there was no such
//  item, does nothing. Returns 0 if successful, else -1.               
CZMQ_EXPORT int
    zhashx_rename (zhashx_t *self, const void *old_key, const void *new_key);

//  Set a free function for the specified hash table item. When the item is
//  destroyed, the free function, if any, is called on that item.          
//  Use this when hash items are dynamically allocated, to ensure that     
//  you don't have memory leaks. You can pass 'free' or NULL as a free_fn. 
//  Returns the item, or NULL if there is no such item.                    
CZMQ_EXPORT void *
    zhashx_freefn (zhashx_t *self, const void *key, zhashx_free_fn free_fn);

//  Return the number of keys/items in the hash table
CZMQ_EXPORT size_t
    zhashx_size (zhashx_t *self);

//  Return a zlistx_t containing the keys for the items in the       
//  table. Uses the key_duplicator to duplicate all keys and sets the
//  key_destructor as destructor for the list.                       
//  Caller owns return value and must destroy it when done.
CZMQ_EXPORT zlistx_t *
    zhashx_keys (zhashx_t *self);

//  Return a zlistx_t containing the values for the items in the  
//  table. Uses the duplicator to duplicate all items and sets the
//  destructor as destructor for the list.                        
//  Caller owns return value and must destroy it when done.
CZMQ_EXPORT zlistx_t *
    zhashx_values (zhashx_t *self);

//  Simple iterator; returns first item in hash table, in no given order, 
//  or NULL if the table is empty. This method is simpler to use than the 
//  foreach() method, which is deprecated. To access the key for this item
//  use zhashx_cursor(). NOTE: do NOT modify the table while iterating.   
CZMQ_EXPORT void *
    zhashx_first (zhashx_t *self);

//  Simple iterator; returns next item in hash table, in no given order, 
//  or NULL if the last item was already returned. Use this together with
//  zhashx_first() to process all items in a hash table. If you need the 
//  items in sorted order, use zhashx_keys() and then zlistx_sort(). To  
//  access the key for this item use zhashx_cursor(). NOTE: do NOT modify
//  the table while iterating.                                           
CZMQ_EXPORT void *
    zhashx_next (zhashx_t *self);

//  After a successful first/next method, returns the key for the item that
//  was returned. This is a constant string that you may not modify or     
//  deallocate, and which lasts as long as the item in the hash. After an  
//  unsuccessful first/next, returns NULL.                                 
CZMQ_EXPORT const void *
    zhashx_cursor (zhashx_t *self);

//  Add a comment to hash table before saving to disk. You can add as many   
//  comment lines as you like. These comment lines are discarded when loading
//  the file. If you use a null format, all comments are deleted.            
CZMQ_EXPORT void
    zhashx_comment (zhashx_t *self, const char *format, ...);

//  Save hash table to a text file in name=value format. Hash values must be
//  printable strings; keys may not contain '=' character. Returns 0 if OK, 
//  else -1 if a file error occurred.                                       
CZMQ_EXPORT int
    zhashx_save (zhashx_t *self, const char *filename);

//  Load hash table from a text file in name=value format; hash table must 
//  already exist. Hash values must printable strings; keys may not contain
//  '=' character. Returns 0 if OK, else -1 if a file was not readable.    
CZMQ_EXPORT int
    zhashx_load (zhashx_t *self, const char *filename);

//  When a hash table was loaded from a file by zhashx_load, this method will
//  reload the file if it has been modified since, and is "stable", i.e. not 
//  still changing. Returns 0 if OK, -1 if there was an error reloading the  
//  file.                                                                    
CZMQ_EXPORT int
    zhashx_refresh (zhashx_t *self);

//  Serialize hash table to a binary frame that can be sent in a message.
//  The packed format is compatible with the 'dictionary' type defined in
//  http://rfc.zeromq.org/spec:35/FILEMQ, and implemented by zproto:     
//                                                                       
//     ; A list of name/value pairs                                      
//     dictionary      = dict-count *( dict-name dict-value )            
//     dict-count      = number-4                                        
//     dict-value      = longstr                                         
//     dict-name       = string                                          
//                                                                       
//     ; Strings are always length + text contents                       
//     longstr         = number-4 *VCHAR                                 
//     string          = number-1 *VCHAR                                 
//                                                                       
//     ; Numbers are unsigned integers in network byte order             
//     number-1        = 1OCTET                                          
//     number-4        = 4OCTET                                          
//                                                                       
//  Comments are not included in the packed data. Item values MUST be    
//  strings.                                                             
//  Caller owns return value and must destroy it when done.
CZMQ_EXPORT zframe_t *
    zhashx_pack (zhashx_t *self);

//  Make a copy of the list; items are duplicated if you set a duplicator 
//  for the list, otherwise not. Copying a null reference returns a null  
//  reference. Note that this method's behavior changed slightly for CZMQ 
//  v3.x, as it does not set nor respect autofree. It does however let you
//  duplicate any hash table safely. The old behavior is in zhashx_dup_v2.
//  Caller owns return value and must destroy it when done.
CZMQ_EXPORT zhashx_t *
    zhashx_dup (zhashx_t *self);

//  Set a user-defined deallocator for hash items; by default items are not
//  freed when the hash is destroyed.                                      
CZMQ_EXPORT void
    zhashx_set_destructor (zhashx_t *self, zhashx_destructor_fn destructor);

//  Set a user-defined duplicator for hash items; by default items are not
//  copied when the hash is duplicated.                                   
CZMQ_EXPORT void
    zhashx_set_duplicator (zhashx_t *self, zhashx_duplicator_fn duplicator);

//  Set a user-defined deallocator for keys; by default keys are freed
//  when the hash is destroyed using free().                          
CZMQ_EXPORT void
    zhashx_set_key_destructor (zhashx_t *self, zhashx_destructor_fn destructor);

//  Set a user-defined duplicator for keys; by default keys are duplicated
//  using strdup.                                                         
CZMQ_EXPORT void
    zhashx_set_key_duplicator (zhashx_t *self, zhashx_duplicator_fn duplicator);

//  Set a user-defined comparator for keys; by default keys are
//  compared using strcmp.                                     
CZMQ_EXPORT void
    zhashx_set_key_comparator (zhashx_t *self, zhashx_comparator_fn comparator);

//  Set a user-defined comparator for keys; by default keys are
//  compared using strcmp.                                     
CZMQ_EXPORT void
    zhashx_set_key_hasher (zhashx_t *self, zhashx_hash_fn hasher);

//  Make copy of hash table; if supplied table is null, returns null.    
//  Does not copy items themselves. Rebuilds new table so may be slow on 
//  very large tables. NOTE: only works with item values that are strings
//  since there's no other way to know how to duplicate the item value.  
CZMQ_EXPORT zhashx_t *
    zhashx_dup_v2 (zhashx_t *self);

//  *** Deprecated method, slated for removal: avoid using it ***
//  Set hash for automatic value destruction. This method is deprecated
//  and you should use set_destructor instead.                         
CZMQ_EXPORT void
    zhashx_autofree (zhashx_t *self);

//  *** Deprecated method, slated for removal: avoid using it ***
//  Apply function to each item in the hash table. Items are iterated in no
//  defined order. Stops if callback function returns non-zero and returns 
//  final return code from callback function (zero = success). This method 
//  is deprecated and you should use zhashx_first/_next instead.           
CZMQ_EXPORT int
    zhashx_foreach (zhashx_t *self, zhashx_foreach_fn callback, void *argument);

//  Self test of this class.
CZMQ_EXPORT void
    zhashx_test (bool verbose);

#ifdef CZMQ_BUILD_DRAFT_API
//  *** Draft method, for development use, may change without warning ***
//  Same as unpack but uses a user-defined deserializer function to convert
//  a longstr back into item format.                                       
CZMQ_EXPORT zhashx_t *
    zhashx_unpack_own (zframe_t *frame, zhashx_deserializer_fn deserializer);

//  *** Draft method, for development use, may change without warning ***
//  Same as pack but uses a user-defined serializer function to convert items
//  into longstr.                                                            
//  Caller owns return value and must destroy it when done.
CZMQ_EXPORT zframe_t *
    zhashx_pack_own (zhashx_t *self, zhashx_serializer_fn serializer);

#endif // CZMQ_BUILD_DRAFT_API
//  @ignore
CZMQ_EXPORT void
    zhashx_comment (zhashx_t *self, const char *format, ...) CHECK_PRINTF (2);
//  @end


#ifdef __cplusplus
}
#endif

#endif
