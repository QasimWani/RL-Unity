class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        #this algorithm is implemented using Lavenshtein distance
        grid = [ [x for x in range(len(word2) + 1)] for y in range(len(word1) + 1)]        
        
        #index the first column
        for row in range(1, len(word1) + 1):
            grid[row][0] = grid[row -1][0] + 1
            
        
        for r in range(1, len(word1) + 1):
            for c in range(1, len(word2) + 1):
                char1, char2 = word1[r-1], word2[c-1]
                if(char1 == char2):
                    #fetch diagonal
                    grid[r][c] = grid[r-1][c-1]
                else:
                    grid[r][c] = 1 + min(grid[r-1][c-1], grid[r-1][c], grid[r][c-1])
                    
        return grid[-1][-1]
